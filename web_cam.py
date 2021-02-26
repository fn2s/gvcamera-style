import sys, os, click, json, warnings, shlex, cv2
from time import gmtime as t
import subprocess as sp
from skvideo.io import FFmpegWriter
warnings.simplefilter("ignore")
@click.command(help="""
	Shows web camera video stream in screen window \n
	Optionally writes it to mp4 file (if '--fileoutput' option set) \n
	Optionally creates virtual camera device (if '--virtualdevice' option set)\n
	For this option, in advance connect web camera, run 'sudo modprobe v4l2loopback'
    and check virtualdevice number via 'v4l2-ctl --list-devices' """)
@click.option('-d', '--device', type=int, default=0,
		help='number of input camera device /dev/videoN')
@click.option('-vd', '--virtualdevice', type=int,
		help='number of output virtual camera device')
@click.option('-r', '--record', is_flag=True,
		help='file name in directory ./ to write video stream')
@click.option('-m', '--mirror', is_flag=True,
		help='to flip input frames horizontally')
def run_demo(device, virtualdevice, record, mirror):
	devName = '/dev/video'+str(device)
	if not os.path.exists(devName):
		sys.exit("No such web camera device: "+ devName, "!  exiting...")
	fps, w, h = str(30), 640, 480
	cam = cv2.VideoCapture(device)
	if not cam.isOpened():
		sys.exit("Can't open web camera device "+ devName, "!  exiting...")
	ret_val, img = cam.read()
	print("CAMERA ORIGINAL SHAPE",img.shape)
	cam.set(3, w); cam.set(4, h)
	ret_val, img = cam.read()
	print("CAMERA AJUSTED SHAPE",img.shape)
	wName = f'Web Camera {devName}   fps:{fps}  W:{str(w)}  H:{str(h)}'
	cv2.namedWindow(wName, cv2.WINDOW_NORMAL)
	if virtualdevice:
		vdevName = '/dev/video'+str(virtualdevice)
		if not os.path.exists(vdevName):
			sys.exit("No such virtual web camera device: "+ vdevName, "!  exiting...")
		ffmpeg_cmd = f'''ffmpeg -loglevel quiet -threads 1 -r {fps} -s {w}x{h} -pix_fmt bgr24
		 -f rawvideo -an -sn -dn -i pipe: -f v4l2 -s {w}x{h} -pix_fmt yuyv422 /dev/video{virtualdevice}'''
		ffmpeg_process = sp.Popen(shlex.split(ffmpeg_cmd), stdin=sp.PIPE)
	if record:
		dtime = f'-{t().tm_year}-{t().tm_mon}-{t().tm_mday}_{t().tm_hour+3}-{t().tm_min}'
		vDir, vFile = './video/', 'webCam'+ str(device) + dtime +'.mp4'
		vPath = os.path.join(vDir, vFile)
		out = FFmpegWriter(vPath, inputdict={'-r': str(fps),'-s':'{}x{}'.format(w, h),
				   '-pix_fmt': 'bgr24'}, outputdict={'-r': str(fps),'-c:v': 'h264'})
	while True:
		ret_val, img = cam.read()
		if ret_val:
			if mirror:
				img = cv2.flip(img, 1)
			cv2.imshow(wName, img)
			if virtualdevice:
				ffmpeg_process.stdin.write(cv2.flip(img, 1).tobytes())
			if record:
				out.writeFrame(img)
		else:
			print("No input video stream !  waiting...")
		key = cv2.waitKey(1)
		if key == 27: # Esc
			break
	cam.release()
	if virtualdevice:
		ffmpeg_process.stdin.flush()
		ffmpeg_process.stdin.close()
		ffmpeg_process.wait()
	if record:
		out.close()
		print("Video file ", vPath, " wriiten OK")
		print ("fps :", fps, "    W:", w, " H:", h)
	cv2.destroyAllWindows()

def main(): run_demo()

if __name__ == '__main__': main()
