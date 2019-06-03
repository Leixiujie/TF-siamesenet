
import numpy as np
import tensorflow as tf
from PIL import Image
import os



filename = tf.placeholder(tf.string, [], name='filename')
image_file = tf.read_file(filename)
# Decode the image as a JPEG file, this will turn it into a Tensor
image_raw = tf.image.decode_jpeg(image_file)  # 图像解码成矩阵
image = 255.0 * tf.image.convert_image_dtype(image_raw, tf.float32)
pics = []
image_ = tf.image.resize_images(image,(224,224),0)
#pics.append(image_)


image1=tf.image.flip_up_down(image) #将图像从上至下顺时针翻转180°
pics.append(image1)


#产生灰度图
image_gray = tf.image.rgb_to_grayscale(image)
image_gray = tf.image.grayscale_to_rgb(image_gray)
pics.append(image_gray)

'''
#生成均值为0的图
adjusted = tf.image.per_image_standardization(image_raw)
adjusted_image = 255.0 * tf.image.convert_image_dtype(adjusted, tf.float32)
pics.append(adjusted_image)
'''
#生成不同亮度的图
delta = -0.3
while(delta <= 0.5):
    image2 = tf.image.adjust_brightness(image_raw,delta)
    delta += 0.3
    image2 = 255.0 * tf.image.convert_image_dtype(image2, tf.float32)
    pics.append(image2)

#生成不同剪切范围的图
crop_size = 120
while(crop_size < 160):
    random_size = int((np.random.randint(0,10)*1.0/15) * crop_size)
    croped = tf.image.resize_image_with_crop_or_pad(image_,crop_size,crop_size+random_size)
    croped = tf.image.resize_images(croped,(224,224),0)
    pics.append(croped)   
    
    croped = tf.image.resize_image_with_crop_or_pad(image_,crop_size + random_size,crop_size)
    croped = tf.image.resize_images(croped,(224,224),0)
    pics.append(croped)
    
    croped = tf.image.resize_image_with_crop_or_pad(image_,crop_size,crop_size)
    croped = tf.image.resize_images(croped,(224,224),0)
    pics.append(croped)
    crop_size += 30

#生成不同对比度的图
contrast = 0.3
while(contrast <= 2.5):
    contrast_pic = tf.image.adjust_contrast(image_raw,contrast)
    contrast_pic = 255.0 * tf.image.convert_image_dtype(contrast_pic, tf.float32)
    pics.append(contrast_pic)
    contrast += 0.5
    
#生成不同色相的图
hue = -0.5
while(hue <= 0.5):
    hue_image = tf.image.adjust_hue(image_raw,hue)
    hue_image = 255.0 * tf.image.convert_image_dtype(hue_image, tf.float32)
    pics.append(hue_image)
    hue += 0.4

#生成不同饱和度的图
saturation = -5
while(saturation <= 5):
    saturation_image = tf.image.adjust_saturation(image_raw,saturation)
    saturation_image = 255.0 * tf.image.convert_image_dtype(saturation_image, tf.float32)
    pics.append(saturation_image)
    saturation += 4

def generate(origin,destin_file):
    with tf.Session() as sess:
      for a,b,pic_all in os.walk(origin):
          break
      print(len(pic_all))
      iii = 0
      for pic_name in pic_all:
          if iii % 10.0 == 0:
              print('正在处理第'+str(iii)+'张tf转换图片')
          iii += 1
          pic = os.path.join(origin,pic_name)
          transformed_pics = sess.run(pics,feed_dict={filename:pic}) 
          
          num = 11
          
          if not os.path.exists(destin_file):
              os.makedirs(destin_file)
              
          for pic in transformed_pics:
              picture = np.asarray(pic,dtype=np.uint8)
              image__ = Image.fromarray(picture)
              image__ = image__.resize((224,224))
              #image__.show()
              image__.save(destin_file+str(num)+'_'+(str(pic_name).split('.')[0])+'.jpg')
              num += 1
          
          degree = -90
          picture = np.asarray(transformed_pics[0],dtype=np.uint8)
          image_ = Image.fromarray(picture)
          #用原图所生成-90~90度旋转图片
          while(degree <= 90): 
              image__ = image_.rotate(degree)
              image__ = image__.resize((224,224))
              if degree != 0:
                  image__.save(destin_file+str(num)+'_'+(str(pic_name).split('.')[0]).strip()+'.jpg')
              degree += 180
              num += 1
          
if __name__ == '__main__':
    origin = './train_with_3_channels/'
    destin_file = './softmax_train/'
    
    
    generate(origin,destin_file)