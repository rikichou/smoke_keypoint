import caffe
import cv2
import numpy as np

caffe.set_mode_cpu()

model_def = 'caffe_model/deploy.prototxt'
model_weights = 'caffe_model/deploy.caffemodel'

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

print(net.blobs['data'].data.shape)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', [2,0,1])
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', [2,1,0])

net.blobs['data'].reshape(1,
                          3,
                          300,
                          300)

#img = caffe.io.load_image('images/00002.jpg')
image = cv2.imread('images/00001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = image/255
transformer_img = transformer.preprocess('data',img)

#image = cv2.resize(image_face, (self.insize, self.insize), interpolation=cv2.INTER_LINEAR)
#image = image.astype(np.float32)
#image -= self.mean
#input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).cuda()

net.blobs['data'].data[...] = transformer_img
output = net.forward()
output_pro = output['prob'][0]

print ('predict class is:',output_pro.argmax())
print("prob ", output_pro)