import numpy as np
from mtr.database_queries.general_queries import general_queries as GQ
from mtr.database_queries.photo_retrieval_queries import PhotoRetrievalQueries as PRQ
from mtr.datou.datou_lib import download_photos
from PIL import Image
from mtr.database_queries.portfolio_queries import PortfolioQueries
import mtr.lib.fotonower_api.fotonower_connect as FC
import cv2
import pickle as pkl
import os
import sys

gq = GQ("sql.fotonower.com", "MTRBack", "python", "Gaspaillard02")
pq = PortfolioQueries(gq)
prq = PRQ(gq)


def get_images_by_photo_ids(list_photo_ids):
    list_photo_ids = [str(x) for x in list_photo_ids]
    the_string = ",".join(list_photo_ids)
    photo_urls = prq.select_photos_from_ids(the_string)
    map_pids_path, map_pids_extension, total_photo_id_missing = download_photos([int(y) for y in list_photo_ids], photo_urls, unique_local_name='Ali', verbose=0)

    return map_pids_path

def get_images_by_portfolio_ids(list_portfolio_ids, catch = None):

    if catch is not None and os.path.exists(catch):
        print("reading data")
        res = pkl.load( open(os.path.join(catch, "res.pkl"), "rb") )
    else:
        res = {}
        for pid in list_portfolio_ids:
            this_photo_ids = pq.select_photo_ids(pid, limit=10000)
            this_res = get_images_by_photo_ids(this_photo_ids)
            res.update(this_res)

        if catch is not None:
            os.mkdir(catch)
            os.mkdir(os.path.join(catch, "temp"))
            print("Catching data")
            for i, this_pid in enumerate(list(res.keys())):
                cur_path = res[this_pid]
                new_path = os.path.join(catch, cur_path)
                os.rename(cur_path, new_path)
                res[this_pid] = new_path
                sys.stdout.write("\r" + "processing... " + str(np.floor(i*100/len(list(res.keys())))) + "%")
            pkl.dump(res, open(os.path.join(catch, "res.pkl") ,"wb") )


    return res

class batch_generator():
    def __init__(self, map_pid_paths, buffer = 1000, image_size = [64, 64], one_time_buffer = False):
        self.mpp = map_pid_paths
        self.cur_idx = 0
        self.buf_index = 0
        self.buffer = buffer
        self.pids = list(self.mpp.keys())
        self.image_size = image_size
        self.buffer_images = []
        self.one_time_buffer = one_time_buffer
        self.update_buffer()

    def next_batch(self, batch_size):
        if self.cur_idx + batch_size > self.buffer:
            self.update_buffer()
        next_images = self.buffer_images[self.cur_idx:(self.cur_idx+batch_size)]
        next_images = np.stack(next_images)
        self.cur_idx += batch_size

        return next_images

    def update_buffer(self):
        if not self.one_time_buffer or len(self.buffer_images) == 0:
            next_idxs = [x % len(self.pids) for x in range(self.buf_index,(self.buf_index+self.buffer))]
            next_pids = [self.pids[x] for x in next_idxs]
            next_paths = [self.mpp[x] for x in next_pids]

            self.buffer_images = []
            for i, path in enumerate(next_paths):
                try:
                    this_image_bgr = cv2.imread(path)/255
                    b,g,r = cv2.split(this_image_bgr)       # get b,g,r
                    this_image = cv2.merge([r,g,b])     # switch it to rgb

                    this_resized_image = cv2.resize(this_image, (self.image_size[0], self.image_size[1]) )
                    # print(np.max(this_resized_image))
                    self.buffer_images.append(this_resized_image)
                except:
                    self.buffer_images.append(self.buffer_images[0].copy())
                    print(path)

                sys.stdout.write("\r" + "FEEDING BUFFER... " + str(np.floor(i*100/self.buffer)) + "%")
            sys.stdout.write("\r" + "FEEDING BUFFER... 100%\n")
            self.cur_idx = 0
        else:
            self.cur_idx = 0

