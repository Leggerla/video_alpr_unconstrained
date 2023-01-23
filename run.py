import sys
import cv2
import numpy as np
import pandas as pd
import traceback

import darknet.python.darknet as dn

from src.label import Label, Shape, dknet_label_conversion
from src.utils import crop_region, nms, im2single
from darknet.python.darknet import detect
from src.keras_utils import load_model, detect_lp

def is_standard(number):
  if len(number) != 7:
    return False

  if number[:2].isalpha() and number[2:4].isdigit() and number[4:].isalpha():
    return True
  else:
    return False

def correct_for_i(number):
  number_list = list(number)
  for i, num in enumerate(number_list):
    if num == 'I' or num == '1':
      if i < 2 or i >= 4:
        number_list[i] = 'I'
      else:
        number_list[i] = '1'

  number = ''.join(number_list)
  return number


if __name__ == '__main__':

	try:

		video_path = sys.argv[1]
		output_file = sys.argv[2]

		vehicle_threshold = .5

		vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg = b'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = b'data/vehicle-detector/voc.data'

		# import pdb; pdb.set_trace()
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

		print('Searching for vehicles using YOLO...')

		all_plate_numbers = []

		vidcap = cv2.VideoCapture(video_path)
		success, image = vidcap.read()

		while success:

			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			img = image.copy()

			R, _ = detect(vehicle_net, vehicle_meta, img, thresh=vehicle_threshold)

			R = [r for r in R if r[0] in [b'car', b'bus']]

			print(('\t\t%d cars found' % len(R)))

			if len(R):

				image_plate_numbers = []

				Iorig = image.copy()
				WH = np.array(Iorig.shape[1::-1], dtype=float)

				for i, r in enumerate(R):

					cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
					tl = np.array([cx - w / 2., cy - h / 2.])
					br = np.array([cx + w / 2., cy + h / 2.])
					label = Label(0, tl, br)
					Icar = crop_region(Iorig, label).astype(np.uint8)

					ratio = float(max(Icar.shape[:2])) / min(Icar.shape[:2])
					side = int(ratio * 288.)
					bound_dim = min(side + (side % (2 ** 4)), 608)
					print(("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio)))

					Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Icar), bound_dim, 2 ** 4, (240, 80),
												lp_threshold)

					print(('\t\t%d license plates found' % len(LlpImgs)))

					if len(LlpImgs):
						Ilp = (255 * LlpImgs[0]).astype(np.uint8)
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_RGB2BGR)
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

						s = Shape(Llp[0].pts)

						#####################
						print('Performing OCR...')
						R, (width, height) = detect(ocr_net, ocr_meta, Ilp, thresh=ocr_threshold, nms=None)

						if len(R):

							L = dknet_label_conversion(R, width, height)
							L = nms(L, .45)

							L.sort(key=lambda x: x.tl()[0])
							lp_str = ''.join([chr(l.cl()) for l in L])

							print(('\t\tLP: %s' % lp_str))

							number = correct_for_i(lp_str)
							if number not in image_plate_numbers:
								image_plate_numbers.append(number)

						else:

							print('No characters found')

				if image_plate_numbers:
					image_plate_numbers.sort()
					ipn_standard = [[ipn, is_standard(ipn)] for ipn in image_plate_numbers]
					all_plate_numbers.append([item for sublist in ipn_standard for item in sublist])

			success, image = vidcap.read()

		pd.DataFrame(all_plate_numbers).to_csv(output_file)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)