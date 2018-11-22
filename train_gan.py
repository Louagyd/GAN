import GANBlocks

import tensorflow as tf
import os

server = False
if int(tf.__version__.split(".")[1]) > 7:
    print("you are running it in local")
    os.environ["TRAIN_DATA_ROOT"] = "E:/Datasets"
    os.environ["MODEL_ROOT"] = "Models"
else:
    print("running in server")
    # os.environ["TRAIN_DATA_ROOT"] = "/home/ali/workarea/Vision/training_data/"
    os.environ["MODEL_ROOT"] = "/home/ali/workarea/Vision/gan_models"
    server = True
    import data_ops


import os

spgen = GANBlocks.simple_generator
spdis = GANBlocks.simple_discriminator
dcgen = GANBlocks.DCGAN_generator
dcdis = GANBlocks.DCGAN_discriminator
bedis = GANBlocks.BEGAN_discriminator


catch_root = os.environ["TRAIN_DATA_ROOT"]
model_root = os.environ["MODEL_ROOT"]

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p", "--portfolio_ids", action="store", type="string", dest="pids", default="", help="portfolio ids to generate, you should seperate them by comma example: 15322,62321,31341")
    parser.add_option("-d", "--data_name", action="store", type="string", dest="data_name", default="", help="if you use this option your dataset will be catched with this name. so for other time there is no need to re-download photos. the root is $TRAIN_DATA_ROOT")
    parser.add_option("-m", "--model_name", action="store", type="string", dest="model_name", default="", help="if you use this option your model will be saved after training with this name. the root is " + model_root + " . make sure this root exists.")
    parser.add_option("--image_size", action="store", type="string", dest="im_size", default="64,64", help="size of image. default is 64,64")
    parser.add_option("--gan_type", action="store", type="string", dest="gan_type", default="MIXGAN", help="type of gan, example: GAN, DCGAN, BEGAN, DRAGAN, MIXGAN")

    parser.add_option("--z_len", action="store", type="string", dest="z_len", default="64", help="lenght of latent variable")
    parser.add_option("--batch_size", action="store", type="string", dest="batch_size", default="32", help="batch_size")
    parser.add_option("--num_steps", action="store", type="string", dest="num_steps", default="50000", help="number of training steps, default is 50000")
    parser.add_option("--save_every", action="store", type="string", dest="save_every", default="1000", help="step for saving and showing some results, defaults: 1000")
    parser.add_option("--buffer", action="store", type="string", dest="buffer", default="10000", help="buffer, size for reading images once and making batches from it")
    parser.add_option("--one_time_buffer", action="store", type="string", dest="one_time_buffer", default="0", help="load images for one time. it can increase speed because you only load images once not each time. use this if you have not a big dataset.")

    (options, args) = parser.parse_args()
    pids = options.pids
    otb = int(options.one_time_buffer)
    pids = pids.split(",")
    data_name = options.data_name
    data_name = options.data_name
    if data_name == "":
        data_name = None
    else:
        data_name = os.path.join(catch_root, data_name)
    model_name = options.model_name
    if model_name == "":
        model_name = None
    else:
        model_name = os.path.join(model_root, model_name)

    gan_type = options.gan_type
    if gan_type not in ["GAN", "DCGAN", "BEGAN", "DRAGAN", "MIXGAN"]:
        print("Invalid type of gan, so we use MIXGAN instead")
        gan_type = "MIXGAN"
    buffer = int(options.buffer)
    z_len = int(options.z_len)
    num_steps = int(options.num_steps) + 1
    save_every = int(options.save_every)
    batch_size = int(options.batch_size)
    image_size = options.im_size
    image_size = image_size.split(",")
    image_size = [int(image_size[0]), int(image_size[1])]

    if server:
        import pickle as pkl
        if pids[0] != "":
            res = data_ops.get_images_by_portfolio_ids(pids, catch=data_name)
        else:
            res = pkl.load(open(os.path.join(data_name, "res.pkl"), "rb"))

        data_content = data_ops.batch_generator(res, buffer=buffer, image_size=image_size, one_time_buffer=otb)
    else:
        from ops import read_tfrecord_image_data
        data_content = read_tfrecord_image_data(os.path.join(data_name, "tfrecord"), "validation", resize=image_size)


    train_config_init = {"batch_size":batch_size, "num_steps":num_steps, "z_sd":1, "new_tf":not server, "model_name": model_name, "save_every": save_every}
    print("CREATING MODEL")
    if gan_type == "GAN":
        config = {"beta1": 0.5, "beta2": 0.99}
        train_config = {"learning_rate":0.0001}
        train_config.update(train_config_init)
        test_optims, test_fd = GANBlocks.simpleGAN(spgen, spdis, config, z_len=z_len, image_shape=image_size + [3], minimax=False)
    if gan_type == "DCGAN":
        config = {"beta1": 0.5, "beta2": 0.99}
        train_config = {"learning_rate":0.0001}
        train_config.update(train_config_init)
        test_optims, test_fd = GANBlocks.simpleGAN(dcgen, dcdis, config, z_len=z_len, image_shape=image_size + [3], minimax=False)
    if gan_type == "BEGAN":
        config = {"beta1": 0.5, "beta2": 0.99, "lambda": 0.001, "gamma": 0.75, "dis_iters":5}
        train_config = {"learning_rate":0.001}
        train_config.update(train_config_init)
        test_optims, test_fd = GANBlocks.BEGAN(dcgen, dcdis, config, z_len=z_len, image_shape=image_size + [3])
    if gan_type == "DRAGAN":
        config = {"beta1": 0.5, "beta2": 0.99, "lambda": 10.0, "dis_iters":1}
        train_config = {"learning_rate":0.0002, "DRAGAN":True}
        train_config.update(train_config_init)
        test_optims, test_fd = GANBlocks.DRAGAN(dcgen, dcdis, config, z_len=z_len, image_shape=image_size + [3])
    if gan_type == "MIXGAN":
        config = {"beta1": 0.5, "beta2": 0.99, "lambda": 0.001, "gamma": 0.75, "dis_iters":5}
        train_config = {"learning_rate":[0.001, 0.7, 5000]}
        train_config.update(train_config_init)
        test_optims, test_fd = GANBlocks.testGAN(dcgen, dcdis, bedis, config, z_len=z_len, image_shape=image_size + [3], minimax=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print("TRAINING MODEL")
    GANBlocks.train_gan(sess, test_optims, test_fd, data_content, train_config)





