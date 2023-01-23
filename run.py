import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label import Label, lwrite, Shape, writeShapes, dknet_label_conversion
from os.path import isdir, splitext, basename
from os import makedirs
from src.utils import crop_region, im2single, nms
from src.keras_utils import load_model, detect_lp
from darknet.python.darknet import detect

if __name__ == '__main__':

	try:

		input_dir = sys.argv[1]
		output_dir = sys.argv[2]
		csv_file = '../tmp/output/results.csv'

		bname = splitext(basename(input_dir))[0]

		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		vehicle_net = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

		wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
		wpod_net = load_model(wpod_net_path)
		lp_threshold = .5

		ocr_threshold = .4

		ocr_weights = b'data/ocr/ocr-net.weights'
		ocr_netcfg = b'data/ocr/ocr-net.cfg'
		ocr_dataset = b'data/ocr/ocr-net.data'

		ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		vidcap = cv2.VideoCapture(input_dir)

		if not isdir(output_dir):
			makedirs(output_dir)

		print
		'Searching for vehicles using YOLO...'

		success, Iorig = vidcap.read()
		while success:

			R, _ = detect(vehicle_net, vehicle_meta, Iorig, thresh=vehicle_threshold)

			R = [r for r in R if r[0] in ['car', 'bus']]

			print
			'\t\t%d cars found' % len(R)

			if len(R):

				WH = np.array(Iorig.shape[1::-1], dtype=float)
				Lcars = []
				Icars = []

				for i, r in enumerate(R):
					cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
					tl = np.array([cx - w / 2., cy - h / 2.])
					br = np.array([cx + w / 2., cy + h / 2.])
					label = Label(0, tl, br)
					Icar = crop_region(Iorig, label)

					Lcars.append(label)
					Icars.append(Icar)

					###############

					ratio = float(max(Icar.shape[:2])) / min(Icar.shape[:2])
					side = int(ratio * 288.)
					bound_dim = min(side + (side % (2 ** 4)), 608)
					print(("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio)))

					Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Icar), bound_dim, 2 ** 4, (240, 80),
												lp_threshold)

					if len(LlpImgs):
						Ilp = LlpImgs[0]
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

						s = Shape(Llp[0].pts)

						#todo cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
						#todo writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])


						#####################
						R, (width, height) = detect(ocr_net, ocr_meta, Ilp, thresh=ocr_threshold, nms=None)

						if len(R):

							L = dknet_label_conversion(R, width, height)
							L = nms(L, .45)

							L.sort(key=lambda x: x.tl()[0])
							lp_str = ''.join([chr(l.cl()) for l in L])

							with open('%s/%s_str.txt' % (output_dir, bname), 'w') as f:
								f.write(lp_str + '\n')

							print(('\t\tLP: %s' % lp_str))

						else:

							print('No characters found')


	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
