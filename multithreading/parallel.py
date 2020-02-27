import logging
import os
from time import time
from queue import Queue
from threading import Thread
from functools import partial
from multiprocessing.pool import Pool

from download import setup_download_dir, get_links, download_link

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s %(message)s')
logging.getLogger('requests').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# class DownloadWorker(Thread):
#     def __init__(self, queue):
#         Thread.__init__(self)
#         self.queue = queue

#     def run(self):
#         while True:
#             directory, link = self.queue.get()
#             try:
#                 download_link(directory, link)
#             finally:
#                 self.queue.task_done()

def main():
    ts = time()
    client_id = '83739fed7b44f1a'
    if not client_id:
        raise Exception("Couldnt find imgur client id")
        
    download_dir = setup_download_dir()
    links = get_links(client_id)
    # create a queue to communicate with worker threads
    # queue = Queue()
    # # create 8 worker threads
    # for x in range(8):
    #     worker = DownloadWorker(queue)
    #     # settig deamon to True will let the thread exit even though the workers are blocking
    #     worker.daemon = True
    #     worker.start()
    # for link in links:
    #     # logger.info("Queueing {}".format(link))
    #     print("Queueing {}".format(link))
    #     queue.put((download_dir, link))
    #     # download_link(download_dir, link)
    # # causes the main thread to wait for the queue to finish
    # queue.join()
    download  = partial(download_link, download_dir)
    with Pool(4) as p:
        p.map(download, links)
    logging.info('Took %s seconds', time() - ts)
    print('Took {} seconds'.format(time() - ts))

if __name__ == '__main__':
    main()