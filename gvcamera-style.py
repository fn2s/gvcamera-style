import sys, os, argparse, json, warnings, shlex, datetime, cv2
from tkinter import *
from PIL import Image, ImageTk
import subprocess as sp
from skvideo.io import FFmpegWriter
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from net import Net
warnings.filterwarnings("ignore")

class St_app:
	def __init__(self, args):
		FPS, HM, WM = 30, 480, 640
		device, self.virtualdevice = args['device'], args['virtualdevice']
		self.record = args['record']
		if args['cuda']:
			self.ctx = mx.gpu(0); os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
		else:
			self.ctx = mx.cpu(0)
		self.style_folder = 'images/styles/'
		model = 'models/21styles.params'
		devName = '/dev/video'+str(device)
		if not os.path.exists(devName):
			sys.exit("No such web camera device: "+ devName)
		self.cam = cv2.VideoCapture(device)
		if not self.cam.isOpened():
			sys.exit("Can't open web camera device "+ devName)
		ret_val, img = self.cam.read()
		while not ret_val:
			print("No input video stream !  waiting...", end='')
		demo_size = 480	# frame height
		fps, w, h = str(FPS), img.shape[1], img.shape[0]
		print("Input Video Original Shape:", h,"x", w)
		self.downsize = w > WM
		if self.downsize :
			h = 2 * int(h * WM / w / 2); w = WM
		print("Input Video Adjusted Shape:", h,"x", w)
		self.fps, self.w, self.h = fps, w, h
		if self.virtualdevice:
			vdevName = '/dev/video'+str(self.virtualdevice)
			if not os.path.exists(vdevName):
				sys.exit("No such virtual web camera device: "+ vdevName)
			ffmpeg_cmd = f'''ffmpeg -loglevel quiet -threads 1 -r {fps} -s {w}x{h} -pix_fmt rgb24
				-f rawvideo -an -sn -dn -i pipe: -f v4l2 -s {w}x{h} -pix_fmt yuyv422 /dev/video{self.virtualdevice}'''
			self.ffmpeg_process = sp.Popen(shlex.split(ffmpeg_cmd), stdin=sp.PIPE)
		if self.record:
			ts = datetime.datetime.now()
			dtime = '{}'.format(ts.strftime("%Y-%m-%d_%H-%M"))
			vDir, vFile = './video/', f'webCam{str(device)}{dtime}.mp4'
			self.vPath = os.path.join(vDir, vFile)
			try:
				if not os.path.exists(vDir):
					os.makedirs(vDir)
			except OSError as e:  print(e); sys.exit(1)
			self.out = FFmpegWriter(self.vPath, inputdict={'-r': fps,'-s':'{}x{}'.format(2*w, h),
							   '-pix_fmt': 'rgb24'}, outputdict={'-r': fps,'-c:v': 'h264'})
		wName = f'STYLIZED VIDEO from CAMERA {devName}   W: {str(w)}  H: {str(h)}'
		self.window = Tk()
		self.window.wm_title(wName)
		self.window.config(background="#FFFFFF")
		self.window.resizable(False, False)
		self.window.protocol('WM_DELETE_WINDOW', self.st_close)

		self.stylebar = Frame(self.window, width=2*w, relief='raised')
		self.stylebar.grid(row=0, column=0, padx=10, pady=2)
		self.files = os.listdir(self.style_folder)
		st_num = len(self.files)
		assert(st_num > 0)
		self.icon = []
		for i in range(st_num):
			ic = cv2.imread(os.path.join(self.style_folder, self.files[i]))
			ic = cv2.resize(ic,(55,55), interpolation = cv2.INTER_CUBIC)
			self.icon.append(ImageTk.PhotoImage(image=Image.fromarray(ic)))
			btn = Button(self.stylebar, image=self.icon[i], command=lambda id=i: self.set_style(id))
			btn.pack(side=LEFT, fill=X)

		self.toolbar = Frame(self.window, width=2*w, height=100, bg='white', relief='raised')
		self.toolbar.grid(row=1, column=0, padx=10, pady=2)
		self.no_style = BooleanVar(self.window)
		self.fh, self.fv = BooleanVar(self.window), BooleanVar(self.window)
		self.iter_styles = BooleanVar(self.window)
#		self.st_period = IntVar(self.window)
#		self.st_period.set(10)
		no_style_cbtn = Checkbutton(self.toolbar, text='Not Stylize, Stream AS IS !!!', bg='white', fg='red', variable = self.no_style)
		no_style_cbtn.grid(row=0, column=0, sticky=W, padx=10, pady=2)
		fh_cbtn = Checkbutton(self.toolbar, text='Flip Horizontally', bg='white', variable = self.fh)
		fh_cbtn.grid(row=0, column=1, sticky=W, padx=10, pady=2)
		fv_cbtn = Checkbutton(self.toolbar, text='Flip Vertically', bg='white', variable = self.fv)
		fv_cbtn.grid(row=0, column=2, sticky=W, padx=10, pady=2)
		iter_styles_cbtn = Checkbutton(self.toolbar, text='Iterate thru all Styles with period:', bg='white', variable = self.iter_styles)
		iter_styles_cbtn.grid(row=0, column=3, sticky=E, padx=10, pady=2)
#		self.period_box = Entry(self.toolbar, text='Period of Iteration', bg='white', textvariable=str(self.st_period), width = 4)
#		self.period_box.grid(row=0, column=3, sticky=W, padx=10, pady=2)
		self.period_sl = Scale(self.toolbar, from_=2, to=50, orient=HORIZONTAL, bg='white')
		self.period_sl.set(10)
		self.period_sl.grid(row=0, column=4, sticky=W, pady=2)
		exit_btn = Button(self.toolbar, text='EXIT', command=self.st_close)
		exit_btn.grid(row=0, column=8, sticky=E, padx=10, pady=2)

		imageFrame = Frame(self.window, width=2*w, height=h)
		imageFrame.grid(row=300, column=0, padx=10, pady=2)
		self.display = Label(imageFrame)
		self.display.grid(row=1, column=0, padx=10, pady=2)
		self.style_model = Net(ngf=128)
		self.style_model.load_parameters(model, ctx=self.ctx)

		self.no_style.set(False), self.fh.set(False), self.fv.set(False), self.iter_styles.set(False)
		self.set_style(14)
		self.idx = 0
		self.st_loop()

	def set_style(self,i):
		self.id = i
		fp = os.path.join(self.style_folder, self.files[self.id])
		style_v = cv2.imread(fp)
		style_v = cv2.cvtColor(style_v, cv2.COLOR_BGR2RGB)
		style_v = cv2.resize(style_v,(512, 512), interpolation = cv2.INTER_CUBIC)
		style_v = np.array(style_v).transpose(2, 0, 1).astype(float)
		style_v = F.expand_dims(mx.nd.array(style_v, ctx=self.ctx), 0)
		self.stimg = np.squeeze(style_v.asnumpy()).transpose(1, 2, 0).astype('uint8')
		self.style_model.set_target(style_v)

	def st_loop(self):
		ret_val, img = self.cam.read()
		if ret_val:
			if self.downsize :
				img = cv2.resize(img, (self.w, self.h), interpolation = cv2.INTER_AREA)
			if self.fh.get():
				img = cv2.flip(img, 1)
			if self.fv.get():
				img = cv2.flip(img, 0)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cimg = img.copy()
			if not self.no_style.get():
				if self.iter_styles.get():
	#				self.st_period = int(self.period_box.get())
					self.st_period = self.period_sl.get()
					self.idx += 1
					if self.idx % self.st_period == 1:
						self.id += 1
						self.set_style(int(self.id % len(self.files)))
				else:
					self.idx = self.id
				img = np.array(img).transpose(2, 0, 1).astype(float)
				img = F.expand_dims(mx.nd.array(img, ctx=self.ctx), 0)
				img = self.style_model(img)
				img = F.clip(img[0], 0, 255).asnumpy()
				img = img.transpose(1, 2, 0).astype('uint8')
			swidth = int(self.w/4); sheight = int(self.h/4)
			simg = cv2.resize(self.stimg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
			simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
			cimg[0:sheight,0:swidth,:]=simg
			fimg = np.concatenate((cimg, img), axis=1)
			label = self.files[self.id].rsplit('.', 1)[0]
			font = cv2.FONT_HERSHEY_SIMPLEX
			textSize = cv2.getTextSize(label, font, 1, 2)[0]
			cv2.putText(fimg, str(self.id+1), (self.w-45, 28), font, 1, (10, 0, 255), 2)
			cv2.putText(fimg, label, (self.w-textSize[0]-60, 28), font, 1, (10, 255, 0), 2)
			imgtk = ImageTk.PhotoImage(image=Image.fromarray(fimg))
			self.display.imgtk = imgtk
			self.display.configure(image=imgtk)
			self.window.after(10, self.st_loop)
			if self.virtualdevice:
				self.ffmpeg_process.stdin.write(cv2.flip(img, 1).tobytes())
			if self.record:
				self.out.writeFrame(fimg)
		else:
			print("No input video stream !")

	def st_close(self):
		print("[INFO] closing...")
		self.window.destroy()
		self.cam.release()
		if self.virtualdevice:
			self.ffmpeg_process.stdin.flush()
			self.ffmpeg_process.stdin.close()
			self.ffmpeg_process.wait()
		if self.record:
			self.out.close()
			print("Video file ", self.vPath, " wriiten OK")
			print ("fps :", self.fps, "    W:", self.w, " H:", self.h)

ap = argparse.ArgumentParser(description="""
    Takes stream from camera dev/video{device} and stylizes it according to pretrained MXNet model.
    By default it's 21-styles model from  https://github.com/StacyYang/MXNet-Gluon-Style-Transfer \n
    Screen and file output stream combines original & stylized video as well as style image
    Optionally stylized video can be streamed to virtual camera dev/video{virtualdevice}
    For this option, in advance connect web camera, run 'sudo modprobe v4l2loopback'
    and check virtualdevice number via 'v4l2-ctl --list-devices'
    Press ESC key to stop  """)
ap.add_argument('-d', '--device', type=int, default=0,
		help='number of input camera device /dev/video<N>')
ap.add_argument('-vd', '--virtualdevice', type=int,
		help='number of output virtual camera device')
ap.add_argument('-c', '--cuda', action='store_true', help='to use CUDA GPU, by default uses CPU')
ap.add_argument('-r', '--record', action='store_true', help='to write video to "video/output21.mp4" file ')

args = vars(ap.parse_args())

sta = St_app(args)
sta.window.mainloop()
