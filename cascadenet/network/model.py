import lasagne
import cascadenet.network.layers as l
from collections import OrderedDict
import tensorflow as tf
from cascadenet.network.layers.fourier import FFT2Layer, FFTCLayer

# def cascade_resnet(pr, net, input_layer, n=5, nf=64, b=lasagne.init.Constant, **kwargs):
#     shape = lasagne.layers.get_output_shape(input_layer)
#     n_channel = shape[1]
#     net[pr+'conv1'] = l.Conv(input_layer, nf, 3, b=b(), name=pr+'conv1')

#     for i in range(2, n):
#         net[pr+'conv%d'%i] = l.Conv(net[pr+'conv%d'%(i-1)], nf, 3, b=b(),
#                                     name=pr+'conv%d'%i)

#     net[pr+'conv_aggr'] = l.ConvAggr(net[pr+'conv%d'%(n-1)], n_channel, 3,
#                                      b=b(), name=pr+'conv_aggr')
#     net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer],
#                                     name=pr+'res')
#     output_layer = net[pr+'res']
#     return net, output_layer
def cascade_resnet(pr, net, input_layer,shape, n=5, nf=64, **kwargs):
    n_channel = shape[3]
    net[pr+'conv1'] = tf.layers.conv2d(inputs=input_layer, filters=nf, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name=pr+'conv1')
    for i in range(2,n):
        net[pr+'conv%d'%i] = tf.layers.conv2d(inputs=net[pr+'conv%d'%(i-1)], filters=nf, kernel_size=[3, 3], padding="same", 
                                                activation=tf.nn.relu, name=pr+'conv%d'%i)

    net[pr+'conv_aggr'] = tf.layers.conv2d(inputs=net[pr+'conv%d'%(n-1)], filters=n_channel, kernel_size=[3, 3], padding="same",
                                             activation=tf.nn.relu, name=pr+'conv_aggr')

    #ResidualLayer
    net[pr+'res']=tf.add(net[pr+'conv_aggr'],input_layer, name=pr+'res')
    output_layer = net[pr+'res']
    return net, output_layer

def cascade_resnet_3d_avg(pr, net, input_layer, n=5, nf=64,
                          b=lasagne.init.Constant, frame_dist=range(5),
                          **kwargs):
    shape = lasagne.layers.get_output_shape(input_layer)
    n_channel = shape[1]
    divide_by_n = kwargs['cascade_i'] != 0
    k = (3, 3, 3)

    # Data sharing layer
    net[pr+'kavg'] = l.AverageInKspaceLayer([input_layer, net['mask']],
                                            shape,
                                            frame_dist=frame_dist,
                                            divide_by_n=divide_by_n,
                                            clipped=False)
    # Conv layers
    net[pr+'conv1'] = l.Conv3D(net[pr+'kavg'], nf, k, b=b(), name=pr+'conv1')

    for i in range(2, n):
        net[pr+'conv%d'%i] = l.Conv3D(net[pr+'conv%d'%(i-1)], nf, k, b=b(),
                                      name=pr+'conv%d'%i)

    net[pr+'conv_aggr'] = l.Conv3DAggr(net[pr+'conv%d'%(n-1)], n_channel, k,
                                       b=b(), name=pr+'conv_aggr')
    net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer],
                                    name=pr+'res')
    output_layer = net[pr+'res']
    return net, output_layer

def DCLayer(incomings,data_shape,inv_noise_level):
    data, mask, sampled = incomings
    data = tf.cast(data,tf.complex64)
    dft2 = tf.fft2d(data, name='dc_dft2')
    dft2 = tf.cast(data,tf.float32)
    x = dft2
    if inv_noise_level:  # noisy case
        out = (x+ v * sampled) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * x + sampled
    

    out = tf.cast(out,tf.complex64)
    idft2 = tf.ifft2d(out, name='dc_idft2')
    idft2 = tf.cast(idft2,tf.float32)
    return idft2

# def build_cascade_cnn_from_list(shape, net_meta, lmda=None):
#     """
#     Create iterative network with more flexibility

#     net_meta: [(model1, cascade1_n),(model2, cascade2_n),....(modelm, cascadem_n),]
#     """
#     if not net_meta:
#         raise

#     net = OrderedDict()
#     input_layer, kspace_input_layer, mask_layer = l.get_dc_input_layers(shape)
#     net['input'] = input_layer
#     net['kspace_input'] = kspace_input_layer
#     net['mask'] = mask_layer

#     j = 0
#     for cascade_net, cascade_n in net_meta:
#         # Cascade layer
#         for i in range(cascade_n):
#             pr = 'c%d_' % j
#             net, output_layer = cascade_net(pr, net, input_layer,
#                                             **{'cascade_i': j})

#             # add data consistency layer
#             net[pr+'dc'] = l.DCLayer([output_layer,
#                                       net['mask'],
#                                       net['kspace_input']],
#                                      shape,
#                                      inv_noise_level=lmda)
#             input_layer = net[pr+'dc']
#             j += 1

#     output_layer = input_layer
#     return net, output_layer

def build_cascade_cnn_from_list(shape, net_meta, lmda=None):
    """
    Create iterative network with more flexibility

    net_meta: [(model1, cascade1_n),(model2, cascade2_n),....(modelm, cascadem_n),]
    """
    if not net_meta:
        raise

    net = OrderedDict()
    #net config with 3 entries: input, kspace_input, mask
    #sess = tf.Session()
    input_layer = tf.placeholder('float', shape, name='input')
    kspace_input_layer = tf.placeholder('float', shape, name='kspace_input')
    mask_layer = tf.placeholder('float', shape, name='mask')
    net['input'] = input_layer
    net['kspace_input'] = kspace_input_layer
    net['mask'] = mask_layer
    j = 0
    for cascade_net, cascade_n in net_meta:
        # Cascade layer
        for i in range(cascade_n):
            pr = 'c%d_' % j
            net, output_layer = cascade_net(pr, net, input_layer,shape=shape, **{'cascade_i': j})

            # add data consistency layer
            net[pr+'dc'] = DCLayer([output_layer,net['mask'],net['kspace_input']],data_shape=shape,inv_noise_level=lmda)
            input_layer = net[pr+'dc']
            j += 1

    output_layer = input_layer
    return net, output_layer




def build_d2_c2(shape):
    def cascade_d2(pr, net, input_layer, **kwargs):
        return cascade_resnet(pr, net, input_layer, shape=shape, n=2)
    return build_cascade_cnn_from_list(shape, [(cascade_d2, 2)])


def build_d5_c5(shape):
    return build_cascade_cnn_from_list(shape, [(cascade_resnet, 5)])


def build_d2_c2_s(shape):
    def cascade_d2(pr, net, input_layer, **kwargs):
        return cascade_resnet_3d_avg(pr, net, input_layer, n=2, nf=16,
                                     frame_dist=range(2), **kwargs)
    return build_cascade_cnn_from_list(shape, [(cascade_d2, 2)])


def build_d5_c10_s(shape):
    return build_cascade_cnn_from_list(shape, [(cascade_resnet_3d_avg, 10)])
