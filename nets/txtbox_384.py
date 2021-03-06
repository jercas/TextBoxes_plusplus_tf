"""
This framework is based on SSD_tensorlow(https://github.com/balancap/SSD-Tensorflow)
"""
import math
import numpy as np
import tensorflow as tf
import tf_extended as tfe
from nets import custom_layers
from nets import textbox_common
from collections import namedtuple
import tensorflow.contrib.slim as slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
# Define a namedtuple data structure to keep the params of TextboxNet class.
TextboxParams = namedtuple('TextboxParameters',['img_shape',
												'num_classes',
												'feat_layers',
												'feat_shapes',
												'anchor_ratios',
												'anchor_sizes',
												'anchor_steps', #辅助放缩搜索网格中心点的位置
												'normalizations',
												'prior_scaling'
])


class TextboxNet(object):
	"""
	Implementation of the Textbox 384 network.
	The default features layers with 384x384 image input are:
	  conv4  ==> 48 x 48
	  conv7  ==> 24 x 24
	  conv8  ==> 12 x 12
	  conv9  ==> 6 x 6
	  conv10 ==> 4 x 4
	  conv11 ==> 2 x 2
	The default image size used to train this network is 384x384.
	Stage 1: 384x384， Stage 2: 768x768.
	"""
	default_params = TextboxParams(
		img_shape=(384, 384),
		num_classes=2,
		feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11'], # 要进行检测的特征图name
		feat_shapes=[(48, 48), (24, 24), (12, 12), (6, 6), (4, 4), (2, 2)], # 特征图大小
		# Anchors 48 = 8 ratios * 6 sizes, each textbox layer 8 = 8 ratios * 1 size
		#TODO: Different anchor ratios setting from the original paper (1,2,3,5,1/2,1/3,1/5)->(2,1/2,3,1/3,4,1/4,5,1/5).
		anchor_ratios=[
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5],
			[2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5],
		], # 特征图先验框所采用的长宽比
		anchor_sizes=[
			(30.,60.),
			(30.,90.),
			(90.,150.),
			(150., 210.),
			(210., 270.),
			(270., 330.)
		], # 不同特征图的先验框尺度（第一个值是s_k，第2个值是s_k+1）
		# 辅助放缩搜索网格中心点的位置
		anchor_steps=[8, 16, 32, 64, 100, 300], # 特征图单元大小 （image_shape / feature_map_shape），用来计算单元的center,
												# 即该层前所有卷积步长stride的乘积， 48*8 = 24*16 = 32*12 = 64*6 = 384 image_size
		normalizations=[20, -1, -1, -1, -1, -1], # L2 Normalization
		prior_scaling=[0.1, 0.1, 0.2, 0.2])  # offset calculate variances


	def __init__(self, params=None):
		"""
		Init the Textbox net with some parameters. Use the default ones
		if none provided.
		"""
		if isinstance(params, TextboxParams):
			self.params = params
		else:
			self.params = self.default_params


	# ======================================================================= #
	def net(self,
			inputs,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_384',
			update_feat_shapes=False):
		"""
		Text network definition.
		"""
		# r = [predictions, localisations, logits, end_points, shape_list(if update_feat_shapes = True)]
		r = text_net(
			inputs,
			feat_layers=self.params.feat_layers,
			anchor_sizes=self.params.anchor_sizes,
			anchor_ratios=self.params.anchor_ratios,
			normalizations=self.params.normalizations,
			is_training=is_training,
			dropout_keep_prob=dropout_keep_prob,
			reuse=reuse,
			scope=scope,
			update_feat_shapes=update_feat_shapes)
		# Update feature shapes (try at least!)
		# feat_shape: [(48, 48), (24, 24), (12, 12), (6, 6), (4, 4), (2, 2)]
		if update_feat_shapes:
			#TODO: r[0]：各选中层预测结果，predictions of r[-1]: 各选中层的feature shape?
			# 获取各个中间层shape（不含0维），如果含有None则返回默认的feat_shapes
			shapes = textboxes_feat_shapes_from_net(r[-1],
													self.params.feat_shapes)
			self.params = self.params._replace(feat_shapes=shapes)
		print(self.default_params.feat_shapes)
		return r


	def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Get the defined Network arg_scope Encapsulation.
		"""
		return ssd_arg_scope(weight_decay, data_format=data_format)


	def arg_scope_caffe(self, caffe_scope):
		"""Get the defined Caffe arg_scope Encapsulation used for weights importing.
		"""
		return ssd_arg_scope_caffe(caffe_scope)

	# ======================================================================= #


	def anchors(self, img_shape, dtype=np.float32):
		"""Compute the default anchor boxes, given an image shape.
		"""
		return textbox_anchor_all_layers(
			img_shape,
			self.params.feat_shapes,
			self.params.anchor_ratios,
			self.params.anchor_sizes,
			self.params.anchor_steps,
			0.5, dtype)


	def bboxes_encode(self,
					  glabels,
					  bboxes,
					  anchors,
					  gxs,
					  gys,
					  scope='text_bboxes_encode'):
		"""Encode labels and bounding boxes.
		"""
		return textbox_common.tf_text_bboxes_encode(
			glabels,
			bboxes,
			anchors,
			gxs,
			gys,
			matching_threshold=0.5,
			prior_scaling=self.params.prior_scaling,
			dtype=tf.float32,
			scope=scope)


	def losses(self,
			   logits,
			   localisations,
			   glabels,
			   glocalisations,
			   gscores,
			   match_threshold=0.5,
			   negative_ratio=3.,
			   alpha=0.2,
			   label_smoothing=0.,
			   batch_size=16,
			   scope='txt_losses'):
		"""Define the SSD network losses.
		"""
		return ssd_losses(
			logits,
			localisations,
			glabels,
			glocalisations,
			gscores,
			match_threshold=match_threshold,
			negative_ratio=negative_ratio,
			alpha=alpha,
			label_smoothing=label_smoothing,
			batch_size=batch_size,
			scope=scope)


def textboxes_feat_shapes_from_net(predictions, default_shapes=None):
	"""Try to obtain the feature shapes from the prediction layers.
	Args:
	  predictions: a list of feature shapes return from text_net()
	Return:
	  list of feature shapes. Default values if predictions shape not fully determined.
	"""
	feat_shapes = []
	for l in predictions:
		print(l)
		shape = l.get_shape().as_list()[1:3]
		# shape = tuple(l[1:3])

		if None in shape:
			return default_shapes
		else:
			feat_shapes.append(shape)
	return feat_shapes


def text_multibox_layer(layer,
						inputs,
						anchor_size,
						anchor_ratio,
						normalization=-1):
	"""
	Construct a multibox layer, return a class and localization predictions.
	The most different between textbox and ssd is the prediction shape
	where textbox has prediction score shape (48,48,2,6) and location has shape (48,48,2,6,12)
	besise,the kernel for every layer same
	"""
	net = inputs
	if normalization > 0:
		net = custom_layers.l2_normalization(net, scaling=True)

	num_classes = 2
	# Number of anchors, 8+2.
	num_anchors = len(anchor_ratio) + len(anchor_size)
	# Vertical offsets.
	num_prior_per_location = num_anchors * 2

	# Location prediction.
	# Location 4+8 coordinates, [minimum horizontal bounding box offsets-4] + [quadrilateral bounding box offsets-8] = 12.
	loc_pred = slim.conv2d(net, num_prior_per_location*12, [3, 5], activation_fn=None,    # 3x5 conv used in textbox layer.
						   padding='SAME', scope='conv_loc')
	# TODO： different from ssd300
	# transform to NHWC
	#loc_pred = custom_layers.channel_to_last(loc_pred)
	loc_pred = slim.flatten(loc_pred)
	l_shape = loc_pred.shape
	batch_size = l_shape[0]
	loc_pred = tf.reshape(loc_pred, [batch_size, -1, 12])
	# NHW(num_anchors+4) --> [NHW, num_anchors, 4]
	#loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred, 4)[:-1]+[num_prior_per_location, 12])

	# Class prediction.
	cls_pred = slim.conv2d(net, num_prior_per_location*num_classes, [3, 5], activation_fn=None,
						   padding='SAME',scope='conv_cls')
	# transform to NHWC
	#cls_pred = custom_layers.channel_to_last(cls_pred)
	l_shape = tf.shape(cls_pred)
	cls_pred = slim.flatten(cls_pred)
	cls_pred = tf.reshape(cls_pred, [batch_size, -1 ,2])
	# NHW(num_anchors+classes) --> [NHW, num_anchors, classes]
	#cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred, 4)[:-1]+[num_prior_per_location, num_classes])

	return cls_pred, loc_pred, l_shape


def tensor_shape(x, rank=3):
	"""Returns the dimensions of a tensor.
	Args:
	  x: A N-D Tensor of shape.
	  rank:
	Returns:
	  A list of dimensions. Dimensions that are statically known are python
		integers,otherwise they are integer scalar tensors.
	"""
	if x.get_shape().is_fully_defined():
		return x.get_shape().as_list()
	else:
		static_shape = x.get_shape().with_rank(rank).as_list()
		dynamic_shape = tf.unstack(tf.shape(x), rank)
		return [s if s is not None else d
				for s, d in zip(static_shape, dynamic_shape)]


def text_net(inputs,
			 feat_layers=TextboxNet.default_params.feat_layers,
			 anchor_sizes=TextboxNet.default_params.anchor_sizes,
			 anchor_ratios = TextboxNet.default_params.anchor_ratios,
			 normalizations=TextboxNet.default_params.normalizations,
			 is_training=True,
			 dropout_keep_prob=0.5,
			 reuse=None,
			 scope='text_box_384',
			 update_feat_shapes=False):
	"""
	Define the backbone (13 original vgg layers + 10 extra conv layers to extract multi-scale feature maps form as SSD)
	of the textboxes and the neck net -- 6 textbox layers.
	:param inputs: input image size
	:param feat_layers: feature map which connect to the textbox layer
	:param anchor_sizes: multi-scale anchor sizes
	:param anchor_ratios: multi-scale anchor aspect ratios [2.0, 1. / 2, 3.0, 1. / 3, 4.0, 1. / 4, 5., 1. / 5]
	:param normalizations: 
	:param is_training: train or not
	:param dropout_keep_prob: 
	:param reuse: 
	:param scope: 
	:param update_feat_shapes: 
	:return: [predictions, localisations, logits, end_points, shape_list(if update_feat_shapes=True)]
	"""
	# End_points collect relevant activations for external use.
	end_points = {}
	with tf.variable_scope(scope, 'text_box_300', [inputs], reuse=reuse):  # 300*300 384*384
		######################################
		# 前五个 Blocks，首先照搬 VGG16 架构   #
		# 注意这里使用 end_points 标注中间结果  #
		######################################
		# ——————————————————Original VGG-16 blocks (total 13 conv layers)———————————————————————
		# Block 1.
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')  # 300 384
		end_points['conv1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 150
		# Block 2.
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')  # 150 192
		end_points['conv2'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 75
		# Block 3.
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')  # 75 81
		end_points['conv3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 38
		# Block 4.
		end_point = 'conv4'
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')  # 38 40
		end_points[end_point] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 19
		# Block 5.
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')  # 19
		end_points['conv5'] = net
		net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')  # 19  Pooling size 2 -> 3

		####################################
		# 后六个 Blocks，使用额外卷积层      #
		####################################
		# ————————————Additional SSD blocks.——————————————————————
		# Block 6: let's dilate the hell out of it!  dilation -> 6
		net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')  # 19
		end_points['conv6'] = net
		# Block 7: 1x1 conv. Because the fuck.
		end_point = 'conv7'
		net = slim.conv2d(net, 1024, [1, 1], scope='conv7')  # 19
		end_points[end_point] = net

		# Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
		end_point = 'conv8'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
		end_points[end_point] = net  # 10

		end_point = 'conv9'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
		end_points[end_point] = net # 5

		end_point = 'conv10'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
		end_points[end_point] = net  # 3

		end_point = 'conv11'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
		end_points[end_point] = net  #

		######################################
		# 每个中间层 end_points 返回中间结果   #
		# 将各层预测结果存入列表，返回给优化函数 #
		######################################
		# Prediction and localisations layers.
		predictions = []
		logits = []
		localisations = []
		shape_list = []
		# feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11']
		for i, layer in enumerate(feat_layers):
			with tf.variable_scope(layer + '_box'):
				cls, loc, shape = text_multibox_layer(layer,
											 end_points[layer],
											 anchor_sizes[i],
											 anchor_ratios[i],
											 normalizations[i])
			prediction_fn = slim.softmax
			# Prediction of conference and bbox location of each textbox layer.
			predictions.append(prediction_fn(cls))
			logits.append(cls)
			localisations.append(loc)
			shape_list.append(shape)

		if update_feat_shapes is True:
			return predictions, localisations, logits, end_points, shape_list
		else:
			return predictions, localisations, logits, end_points


def textbox_anchor_one_layer(img_shape, feat_shape, ratios, size, step, offset=0.5, dtype=np.float32):
	"""
		Computer TextBoxes++ default anchor boxes for one selected feature layer.
		Each feature point has 12 default textboxes (6 default boxes + 6 offsets vertical boxes).
		Determine the relative position grid of the centers and the relative width and height.

	:param: img_shape: Image shape, used for computing height, width relatively to the former;
	:param: feat_shape: Feature shape of each layer which connect to textbox layer, used for computing relative position grids;
	:param: ratios: Anchor Aspect ratios to use on these features;
	:param: size: Anchor size;
	:param: step:
	:param:offset: Anchor vertical offset.

	:return: y, x, h, w: Relative x and y grids, and height and width of a anchor associated to a textbox layer.
	"""
	# Follow the papers scheme
	# 12 ahchors boxes with out sk' = sqrt(sk * sk+1), 生成feat_shape中HW对应的网格坐标
	y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]] + offset

	# vertical offset, step*feat_shape 约等于img_shape，这使得网格点坐标介于0~1，放缩一下即可到图像大小
	y = y.astype(dtype) * step / img_shape[0]
	x = x.astype(dtype) * step / img_shape[1]
	y_offset = (y.astype(dtype) + offset) * step / img_shape[0]
	x_offset = x

	# (38, 38, 2) origin anchor + offset anchor
	x_out = np.stack((x, x_offset), -1)
	y_out = np.stack((y, y_offset), -1)

	# Expand dims to support easy broadcasting
	y_out = np.expand_dims(y_out, axis=-1)
	x_out = np.expand_dims(x_out, axis=-1)

	# Compute relative height and width, 8 ratios + 2 sizes.
	num_anchors = len(ratios) + len(size)
	# shape: (10,)
	h = np.zeros((num_anchors,), dtype=dtype)
	w = np.zeros((num_anchors,), dtype=dtype)

	# Add first anchor boxes with ratio=1 (smaller square).
	# e.g., h[0] = 30 / 300, w[0] = 30 / 300.
	h[0] = size[0] / img_shape[0]
	w[0] = size[0] / img_shape[1]
	di = 1

	if len(size) > 1:
		# Add last anchor boxes with ratio=1' (bigger square).
		# e.g., h[1] = sqrt(30 * 60) / 300, w[1] = sqrt(30 * 60) / 300.
		h[1] = math.sqrt(size[0] * size[1]) / img_shape[0]
		w[1] = math.sqrt(size[0] * size[1]) / img_shape[1]
		di += 1

	for i, r in enumerate(ratios):
		# Add the other anchors.
		# h[k] = Sk / sqrt(ar), w[k] = sk * sqrt(ar). e.g., h[2] = 30 / 300 / sqrt(2.0), w[2] = 30 / 300 * sqrt(2.0).
		h[i+di] = size[0] / img_shape[0] / math.sqrt(r)
		w[i+di] = size[0] / img_shape[1] * math.sqrt(r)

	xmin = x_out - w/2
	ymin = y_out - h/2
	xmax = x_out + w/2
	ymax = y_out + h/2

	xmin = xmin.reshape([xmin.shape[0], xmin.shape[1], -1], order='F').reshape(-1)
	ymin = ymin.reshape([ymin.shape[0], ymin.shape[1], -1], order='F').reshape(-1)
	xmax = xmax.reshape([xmax.shape[0], xmax.shape[1], -1], order='F').reshape(-1)
	ymax = ymax.reshape([ymax.shape[0], ymax.shape[1], -1], order='F').reshape(-1)

	return xmin, ymin, xmax, ymax


## produce anchor for all layers
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
	"""Defines the VGG arg scope, 约束网络中的超参数设定.

	Args:
	  weight_decay: The l2 regularization coefficient.

	Returns:
	  An arg_scope.
	"""
	with slim.arg_scope(
		[slim.conv2d, slim.fully_connected],
			activation_fn=tf.nn.relu,
			weights_regularizer=slim.l2_regularizer(weight_decay),
			weights_initializer=tf.contrib.layers.xavier_initializer(),
			biases_initializer=tf.zeros_initializer()):
		with slim.arg_scope(
			[slim.conv2d, slim.max_pool2d],
				padding='SAME',
				data_format=data_format):
			with slim.arg_scope(
				[custom_layers.pad2d,
				 custom_layers.l2_normalization,
				 custom_layers.channel_to_last],
				data_format=data_format) as sc:
				return sc


def textbox_anchor_all_layers(img_shape, layers_shape, anchor_ratios, anchor_sizes, anchor_steps,
							  offset=0.5, dtype=np.float32):
	"""
		Compute anchor boxes for all feature layers.
	:param img_shape:
	:param layers_shape:
	:param anchor_ratios:
	:param anchor_sizes:
	:param anchor_steps:
	:param offset: vertical offset which used to cover all text and
			makes the anchors dense in the vertical orientation.
	:param dtype:
	:return: the anchors list of each layer
	"""
	layers_anchors = []
	# layers_shape: [(48, 48), (24, 24), (12, 12), (6, 6), (4, 4), (2, 2)]
	for i, s in enumerate(layers_shape):
		anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
												 anchor_ratios[i],
												 anchor_sizes[i],
												 anchor_steps[i],
												 offset=offset, dtype=dtype)
		layers_anchors.append(anchor_bboxes)
	return layers_anchors


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
	"""Caffe scope definition.

	Args:
	  caffe_scope: Caffe scope object with loaded weights.

	Returns:
	  An arg_scope.
	"""
	# Default network arg scope.
	with slim.arg_scope(
		[slim.conv2d],
			activation_fn=tf.nn.relu,
			weights_initializer=caffe_scope.conv_weights_init(),
			biases_initializer=caffe_scope.conv_biases_init()):
		with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
			with slim.arg_scope(
				[custom_layers.l2_normalization],
					scale_initializer=caffe_scope.l2_norm_scale_init()):
				with slim.arg_scope(
					[slim.conv2d, slim.max_pool2d], padding='SAME') as sc:
					return sc


# =========================================================================== #
# Text loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
			   glabels, glocalisations, gscores,
			   match_threshold=0.5,
			   negative_ratio=3.,
			   alpha=0.2,
			   label_smoothing=0.,
			   batch_size=16,
			   scope=None):
	"""
	Loss functions for training the textboxes++ network.
	This function defines the different loss components of the SSD, and adds them to the TF loss collection.

	Arguments:
	  logits: (list of) predictions logits Tensors;                x
	  localisations: (list of) predictions localisations Tensors;  l
	  glocalisations: (list of) groundtruth localisations Tensors; g
	  gscores: (list of) groundtruth score / conference Tensors;   c
	"""
	# from ssd loss
	with tf.name_scope(scope, 'txt_losses'):
		lshape = tfe.get_shape(logits[0], 5)
		num_classes = lshape[-1]
		batch_size = batch_size

		l_cross_pos = []
		l_cross_neg = []
		l_loc = []

		# Flatten out all vectors!
		flogits = []
		fgscores = []
		flocalisations = []
		fglocalisations = []
		fglabels = []

		# Transform all matrix to vector.
		for i in range(len(logits)):
			flogits.append(tf.reshape(logits[i], [-1, num_classes]))
			fgscores.append(tf.reshape(gscores[i], [-1]))
			fglabels.append(tf.reshape(glabels[i], [-1]))
			flocalisations.append(tf.reshape(localisations[i], [-1, 12]))
			fglocalisations.append(tf.reshape(glocalisations[i], [-1, 12]))

		# And concat the crap!
		glabels = tf.concat(fglabels, axis=0)
		logits = tf.concat(flogits, axis=0)                      # x
		gscores = tf.concat(fgscores, axis=0)                   # c
		localisations = tf.concat(flocalisations, axis=0)      # l
		glocalisations = tf.concat(fglocalisations, axis=0)   # g
		dtype = logits.dtype

		# Compute positive matching mask...
		pmask = gscores > match_threshold  # positive mask
		# pmask = tf.concat(axis=0, values=[pmask[:tf.argmax(gscores, axis=0)], [True], pmask[tf.argmax(gscores, axis=0) + 1:]])
		ipmask = tf.cast(pmask, tf.int32)  # int positive mask
		fpmask = tf.cast(pmask, dtype)  # float positive mask
		n_positives = tf.reduce_sum(fpmask)  # calculate all number

		# Hard negative mining...
		no_classes = tf.cast(pmask, tf.int32)
		predictions = slim.softmax(logits)  #
		nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)  #
		fnmask = tf.cast(nmask, dtype)
		nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
		nvalues_flat = tf.reshape(nvalues, [-1])
		# Number of negative entries to select.
		max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
		n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
		n_neg = tf.minimum(n_neg, max_neg_entries)

		val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
		max_hard_pred = -val[-1]
		# Final negative mask.
		nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
		fnmask = tf.cast(nmask, dtype)

		# Add cross-entropy loss.
		# logits [batch_size, num_classes] labels [batch_size] ~ 0,num_class
		with tf.name_scope('cross_entropy_pos'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits, labels=glabels)
			loss = tf.div(
				tf.reduce_sum(loss * fpmask), batch_size, name='value')
			tf.losses.add_loss(loss)
			l_cross_pos.append(loss)

		with tf.name_scope('cross_entropy_neg'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits, labels=no_classes)
			loss = tf.div(
				tf.reduce_sum(loss * fnmask), batch_size, name='value')
			tf.losses.add_loss(loss)
			l_cross_neg.append(loss)

		# Add localization loss: smooth L1, L2, ...
		with tf.name_scope('localization'):
			# Weights Tensor: positive mask + random negative.
			weights = tf.expand_dims(alpha * fpmask, axis=-1)
			# localisations = tf.Print(localisations, [localisations, tf.shape(localisations)], "pre is:         ", summarize=20)
			# glocalisations = tf.Print(glocalisations, [glocalisations,  tf.shape(glocalisations)], "gt is :         ",summarize=20)
			loss = custom_layers.abs_smooth(localisations - glocalisations)
			loss = tf.div(
				tf.reduce_sum(loss * weights), batch_size, name='value')
			tf.losses.add_loss(loss)
			l_loc.append(loss)