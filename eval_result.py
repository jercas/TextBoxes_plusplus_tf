import os
import time
import tensorflow as tf
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()
model_dir = './model_1steps_384'

def eval_net():
    # ckpt_list = tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
    # for ckpt_path in ckpt_list:
        # if os.path.exists('{}.data-00000-of-00001'.format(ckpt_path)):
    print(time.asctime())
    time.sleep(5)
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    os.system("python test.py -c -0 -m {} -o test_result/{}".format(ckpt_path, ckpt_path.split('/')[-1]))

eval_net()
scheduler.add_job(eval_net, 'interval', seconds=1200)
scheduler.start()
