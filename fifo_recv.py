import os
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from multiprocessing import Queue as MPQueue
from multiprocessing import Process
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
import time
import sys

FIFO_PATH = "/home/yinwenpei/rtc_signal/v_fifo"
FRAME_SIZE = 1382400
width = 1280
height = 720

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log',
                    help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true',
                    help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true',
                    help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()


# convert 1d yuv frame data to rgb matrix
def yuv1d2rgb(yuv1d):
    if yuv1d == None:
        return None
    yuv_matrix = np.array(list(yuv1d)).reshape(int(height * 3 / 2), width)
    yuv_matrix = yuv_matrix.astype('uint8')
    rgb_frame = cv2.cvtColor(yuv_matrix, cv2.COLOR_YUV420p2BGR)
    return rgb_frame


# read_frame from webrtc fifo
def read_frame(read_buffer, fifo_path):
    f_v = os.open(fifo_path, os.O_RDONLY)
    while True:
        read_buffer.put(os.read(f_v, FRAME_SIZE))


# Create fifo if it doesn't exist
if not os.path.exists(FIFO_PATH):
    os.mkfifo(FIFO_PATH)

if os.path.exists("/home/yinwenpei/rtc_signal/filename.txt"):
    os.remove("/home/yinwenpei/rtc_signal/filename.txt")

# fifo frame buffer
read_buffer = MPQueue(-1)

# start read fifo
p_test_read = Process(target=read_frame, args=(read_buffer, FIFO_PATH))
p_test_read.start()

# args.img check, maybe no effect
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

# set cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if (args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

# load model
try:
    try:
        from model.oldmodel.RIFE_HDv2 import Model

        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model

        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v3.x HD model.")
except:
    from model.oldmodel.RIFE_HD import Model

    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v1.x HD model")
model.eval()
model.device()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
print("listening on fifo")
print("read_buffer_size: ", read_buffer.qsize())
lastframe = yuv1d2rgb(read_buffer.get())

# set output parameters
vid_out_name = None
vid_out = None

if args.output is not None:
    vid_out_name = args.output
else:
    vid_out_name = 'out_video.mp4'
vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (width, height))


# inference process
def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])


# def build_read_buffer(user_args, read_buffer, videogen):
#     try:
#         for frame in videogen:
#             if not user_args.img is None:
#                 frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
#             if user_args.montage:
#                 frame = frame[:, left: left + width]
#             read_buffer.put(frame)
#     except:
#         pass
#     read_buffer.put(None)


def make_inference(I0, I1, n):
    global model
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n // 2)
    second_half = make_inference(middle, I1, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def pad_image(img):
    if (args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


if args.montage:
    left = width // 4
    width = width // 2
tmp = max(32, int(32 / args.scale))
ph = ((height - 1) // tmp + 1) * tmp
pw = ((width - 1) // tmp + 1) * tmp
padding = (0, pw - width, 0, ph - height)

if args.montage:
    lastframe = lastframe[:, left: left + width]
write_buffer = Queue(maxsize=500)
# read_buffer = Queue(maxsize=500)
# videogen = []

# _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None # save lastframe when processing static frameni

count = 0
while True:
    print("read_buffer_size: ", read_buffer.qsize())
    if(read_buffer.qsize()==0):
        break
    # if os.path.exists("/home/yinwenpei/rtc_signal/filename.txt"):
    #     read_buffer.put(None)
    # if read_buffer.get() != None:
    #     count += 1
    # else:
    #     break
    if temp is not None:
        frame = temp
        temp = None
    else:
        try:
            frame = yuv1d2rgb(read_buffer.get())
        except:
            frame = None
    if frame is None:
        break
    count += 1
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        try:
            frame = yuv1d2rgb(read_buffer.get())  # read a new frame
            count += 1
        except:
            frame = None
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:height, :width]

    if ssim < 0.2:
        output = []
        for i in range((2 ** args.exp) - 1):
            output.append(I0)
        '''
        output = []
        step = 1 / (2 ** args.exp)
        alpha = 0
        for i in range((2 ** args.exp) - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        '''
    else:
        output = make_inference(I0, I1, 2 ** args.exp - 1) if args.exp else []

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:height, :width])
    # pbar.update(1)
    lastframe = frame
    if break_flag:
        break


if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)

while (not write_buffer.empty()):
    time.sleep(0.1)
# pbar.close()
if not vid_out is None:
    vid_out.release()

p_test_read.kill()

os.remove(FIFO_PATH)
print("count: ", count)
print("all processes ended.")
os._exit(0)

    # tmp = read_buffer.get()
    # yuv_matrix = np.array(list(read_buffer.get())).reshape(int(height * 3 / 2), width)
    # yuv_matrix = yuv_matrix.astype('uint8')
    # rgb = cv2.cvtColor(yuv_matrix, cv2.COLOR_YUV420p2RGB)
    #rgb = read_buffer.get()
    #cv2.imwrite('rgb' + str(count) + '.bmp', rgb)
    # count += 1
