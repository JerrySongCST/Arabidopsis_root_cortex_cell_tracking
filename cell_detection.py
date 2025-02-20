import torch
from torch import cat, einsum, argmax, unsqueeze, FloatTensor
import torch.nn as nn
from torch.nn import init
from scipy.ndimage import zoom
from torch.nn.functional import softmax
from numpy import array, expand_dims, average, sqrt, argwhere, zeros, shape, pad
from skimage import measure, morphology, io
from skimage.color import rgba2rgb, rgb2gray
from os import makedirs
import matplotlib.pyplot as plt

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.outconv1 = nn.Conv2d(filters[0], n_classes, 1, padding=0)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024
        d1 = self.outconv1(up1)  # 256
        return d1

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = int(in_channels/2 if in_channels > out_channels else out_channels/2)
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(int(in_channels/2+in_channels), out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        x = torch.cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class unet3d(nn.Module):
    def __init__(self, init_channels=1, class_nums=3, batch_norm=True, sample=True):
        super(unet3d, self).__init__()
        self.down1 = pub(init_channels, 64, batch_norm)
        self.down2 = unet3dDown(64, 128, batch_norm)
        self.down3 = unet3dDown(128, 256, batch_norm)
        self.down4 = unet3dDown(256, 512, batch_norm)
        self.up3 = unet3dUp(512, 256, batch_norm, sample)
        self.up2 = unet3dUp(256, 128, batch_norm, sample)
        self.up1 = unet3dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.con_last(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def common_mitotic_cell_regions_3d(img):
    img_shape = shape(img)
    images_cells = zeros(img_shape)
    images_mitotic_cells = zeros(img_shape)
    locations = argwhere(img == 2)
    locations_mitotic = argwhere(img == 1)
    for temp in locations:
        images_cells[temp[0], temp[1], temp[2]] = 1
    for temp in locations_mitotic:
        images_mitotic_cells[temp[0], temp[1], temp[2]] = 1
    return images_cells, images_mitotic_cells


def measure_region(mask, connectivity=1):
    label_img, num = measure.label(mask, return_num=True, connectivity=connectivity)
    return label_img, num

def distance_between_points(point1, point2):
    if len(point2) != len(point1):
        print('Error, the length of two points is different')
    else:
        temp = 0
        for i in range(len(point1)):
            d_j = point1[i] - point2[i]
            temp += float(d_j ** 2)
        return sqrt(temp)

def remove_overlapped_cells(all_cells, threshold=4):
    tracked_index = []
    new_all_cells = []
    for i, cell in enumerate(all_cells):
        if i not in tracked_index:
            temp_i = []
            temp_i.append(cell)
            for j, cell_1 in enumerate(all_cells):
                if j not in tracked_index:
                    if i != j:
                        dist = distance_between_points(cell, cell_1)
                        if dist <= threshold:
                            temp_i.append(cell_1)
                            tracked_index.append(j)
            temp_i = average(array(temp_i), 0)
            new_all_cells.append([temp_i[0], temp_i[1], temp_i[2]])
    return new_all_cells



def detect_middle_slice(model, path, idx, slice, device):
    makedirs(f'{path}/images', exist_ok=True)
    slice = slice[0:848, 0:480]
    plt.imsave(r'{0}/images/{1}_middle.png'.format(path, idx), slice)
    data = io.imread(r'{0}/images/{1}_middle.png'.format(path, idx))
    data = rgba2rgb(data)
    data = rgb2gray(data)
    data = expand_dims(data, -1)
    data = data.transpose((2, 0, 1))
    data = expand_dims(data, 0)
    slice_torch = torch.FloatTensor(data)
    pred = model(slice_torch.to(device))
    pred_softmax = softmax(pred, dim=1)
    pred_mask = argmax(pred_softmax, dim=1)
    pred_mask_numpy = pred_mask[0].cpu().data.numpy()
    plt.imsave(r'{0}/images/{1}_middle_mask.png'.format(path, idx), pred_mask_numpy)
    pred_mask_final = morphology.remove_small_objects(array(pred_mask_numpy, dtype=bool), 0.5)
    return pred_mask_final

def pad_image(volume, height=1024, width=512):
    if shape(volume)[1] != height or shape(volume)[2] != width:
        if shape(volume)[1] != height:
            if shape(volume)[2] != width:
                padding_h = int((height-shape(volume)[1])/2)
                padding_x = int((width -shape(volume)[2]) / 2)
                volume = pad(volume, ((0, 0),
                                      (padding_h, height-shape(volume)[1]-padding_h),
                                      (padding_x, width-shape(volume)[2]-padding_x)),
                             'constant')
            else:
                padding_h = int((height - shape(volume)[1]) / 2)
                padding_x = 0
                volume = pad(volume, ((0, 0),
                                      (padding_h, height - shape(volume)[1] - padding_h),
                                      (0, 0)),
                             'constant')
        else:
            padding_x = int((width - shape(volume)[2]) / 2)
            padding_h = 0
            volume = pad(volume, ((0, 0),
                                  (0, 0),
                                  (padding_x, width - shape(volume)[2] - padding_x)),
                         'constant')
    else:
        padding_h = 0
        padding_x = 0

    return volume, padding_h, padding_x

def pred_img(data, model, device):
    data = expand_dims(data, -1)
    data = data.transpose((2, 0, 1))
    data = expand_dims(data, 0)
    slice_torch = torch.FloatTensor(data)
    pred = model(slice_torch.to(device))
    pred_softmax = softmax(pred, dim=1)
    pred_mask = argmax(pred_softmax, dim=1)
    return pred_mask[0].cpu().data.numpy()

def test_model(model, path, idx, imgs, device, whether_transpose):
    h_ratio = shape(imgs)[-2] // 1024
    w_ratio = shape(imgs)[-1] // 512
    temp_masks = []
    common_centroids = []
    mitotic_centroids = []

    if h_ratio > 1 and w_ratio > 1:
        volume = imgs[idx][:, 0:h_ratio * 1024, 0:w_ratio * 512]
        padding_h = padding_x = 0
        for k in range(len(volume)):
            for i in range(h_ratio):
                for j in range(w_ratio):
                    temp = volume[k][i * 1024:(i + 1) * 1024, j * 512:(j + 1) * 512]
                    temp = temp / 255
                    temp_mask = pred_img(temp, model, device)
                    temp_masks.append(temp_mask)
        pred_masks = zeros((len(volume), h_ratio * 1024, w_ratio * 512))
        for k in range(len(volume)):
            for i in range(h_ratio):
                for j in range(w_ratio):
                    pred_masks[k, i * 1024:(i + 1) * 1024, j * 512:(j + 1) * 512] = temp_masks[k * h_ratio * w_ratio + i * w_ratio + j]
    elif h_ratio > 1 and w_ratio <=1:
        volume = imgs[idx][:, 0:h_ratio * 1024, 0:512]
        volume, padding_h, padding_x = pad_image(volume, h_ratio*1024, 512)
        for k in range(len(volume)):
            for i in range(h_ratio):
                temp = volume[k][i * 1024:(i + 1) * 1024, 0:512]
                temp = temp / 255
                temp_mask = pred_img(temp, model, device)
                temp_masks.append(temp_mask)
        pred_masks = zeros((len(volume), h_ratio * 1024, 512))
        for k in range(len(volume)):
            for i in range(h_ratio):
                pred_masks[k, i * 1024:(i + 1) * 1024, 512] = temp_masks[k * h_ratio+i]
    elif h_ratio<=1 and w_ratio >1:
        volume = imgs[idx][:, 0:1024, 0:w_ratio * 512]
        volume, padding_h, padding_x = pad_image(volume, 1024, w_ratio * 512)
        for k in range(len(volume)):
            for j in range(w_ratio):
                temp = volume[k][0:1024, j * 512:(j + 1) * 512]
                temp = temp / 255
                temp_mask = pred_img(temp, model, device)
                temp_masks.append(temp_mask)
        pred_masks = zeros((len(volume), 1024, w_ratio * 512))
        for k in range(len(volume)):
            for j in range(w_ratio):
                pred_masks[k, 1024,  j * 512:(j + 1) * 512] = temp_masks[k * w_ratio + j]
    else:
        volume = imgs[idx][:, 0:1024, 0:512]
        volume, padding_h, padding_x = pad_image(volume)
        for k in range(len(volume)):
            temp = volume[k]
            temp = temp/255
            temp_mask = pred_img(temp, model, device)
            temp_masks.append(temp_mask)
        pred_masks = zeros((len(volume), 1024, 512))
        for k in range(len(volume)):
            pred_masks[k] = temp_masks[k]

    p_c, p_m = common_mitotic_cell_regions_3d(pred_masks)
    p_c = morphology.remove_small_objects(array(p_c, dtype=bool), 1)  # 3.5
    p_m = morphology.remove_small_objects(array(p_m, dtype=bool), 3)  # 50
    region_c, _ = measure_region(p_c)
    region_m, _ = measure_region(p_m)
    common_regions = measure.regionprops(region_c)
    mitotic_regions = measure.regionprops(region_m)

    for props in common_regions:
        z0, y0, x0 = props.centroid
        if whether_transpose:
            common_centroids.append([z0, x0 - padding_x, y0 - padding_h])
        else:
            common_centroids.append([z0, y0 - padding_h, x0 - padding_x])

    for props in mitotic_regions:
        z0, y0, x0 = props.centroid
        if whether_transpose:
            mitotic_centroids.append([z0, x0 - padding_x, y0 - padding_h])
        else:
            mitotic_centroids.append([z0, y0-padding_h, x0-padding_x])
    return remove_overlapped_cells(common_centroids), remove_overlapped_cells(mitotic_centroids)


def test_3d(model, path, idx, imgs, device, whether_transpose):
    h_ratio = shape(imgs)[-2] // 1024
    w_ratio = shape(imgs)[-1] // 512
    temp_masks = []
    common_centroids = []
    mitotic_centroids = []

    if h_ratio > 1 and w_ratio > 1:
        volume = imgs[idx][:, 0:h_ratio * 1024, 0:w_ratio * 512]
        padding_h = padding_x = 0
        for i in range(h_ratio):
            for j in range(w_ratio):
                temp = volume[:, i * 1024:(i + 1) * 1024, j * 512:(j + 1) * 512]
                temp = temp / 255
                temp_mask = pred_img(temp, model, device)
                temp_masks.append(temp_mask)
        pred_masks = zeros((len(volume), h_ratio * 1024, w_ratio * 512))
        for k in range(len(volume)):
            for i in range(h_ratio):
                for j in range(w_ratio):
                    pred_masks[k, i * 1024:(i + 1) * 1024, j * 512:(j + 1) * 512] = temp_masks[
                        k * h_ratio * w_ratio + i * w_ratio + j]
    elif h_ratio > 1 and w_ratio <= 1:
        volume = imgs[idx][:, 0:h_ratio * 1024, 0:512]
        volume, padding_h, padding_x = pad_image(volume, h_ratio * 1024, 512)
        for k in range(len(volume)):
            for i in range(h_ratio):
                temp = volume[k][i * 1024:(i + 1) * 1024, 0:512]
                temp = temp / 255
                temp_mask = pred_img(temp, model, device)
                temp_masks.append(temp_mask)
        pred_masks = zeros((len(volume), h_ratio * 1024, 512))
        for k in range(len(volume)):
            for i in range(h_ratio):
                pred_masks[k, i * 1024:(i + 1) * 1024, 512] = temp_masks[k * h_ratio + i]



    if shape(imgs)[-2] > 1100 or shape(imgs)[-1] > 600:
        if shape(imgs)[-2] > 1100 and shape(imgs)[-1] > 600:
            resize_parameter_h = shape(imgs)[-2]/1100
            resize_parameter_w = shape(imgs)[-1]/600
            crop_h = int(1100 * resize_parameter_h)
            crop_w = int(600 * resize_parameter_w)
            volume = imgs[idx][:, 0:crop_h, 0:crop_w]
            volume = zoom(volume, [1, 1/resize_parameter_h, 1/resize_parameter_w])
            volume = volume[:, 0:1024, 0:512]
        elif shape(imgs)[-2] > 1024:
            resize_parameter_h = shape(imgs)[-2] / 1024
            resize_parameter_w = 1
            crop_h = int(1024 * resize_parameter_h)
            volume = imgs[idx][:, 0:crop_h, :]
            volume = zoom(volume, [1, 1 / resize_parameter_h, 1])
            volume = volume[:, 0:1024, 0:512]
        else:
            resize_parameter_h = 1
            resize_parameter_w = shape(imgs)[-1] / 512
            crop_w = int(512 * resize_parameter_w)
            volume = imgs[idx][:, :, 0:crop_w]
            volume = zoom(volume, [1, 1, 1 / resize_parameter_w])
            volume = volume[:, 0:1024, 0:512]
    else:
        volume = imgs[idx][:, 0:1024, 0:512]
        resize_parameter_h = 1
        resize_parameter_w = 1

    if shape(volume)[1] != 1024 or shape(volume)[2] != 512:
        if shape(volume)[1] != 1024:
            if shape(volume)[2] != 512:
                padding_h = int((1024 - shape(volume)[1]) / 2)
                padding_x = int((512 - shape(volume)[2]) / 2)
                volume = pad(volume, ((0, 0),
                                      (padding_h, 1024 - shape(volume)[1] - padding_h),
                                      (padding_x, 512 - shape(volume)[2] - padding_x)),
                             'constant')
            else:
                padding_h = int((1024 - shape(volume)[1]) / 2)
                padding_x = 0
                volume = pad(volume, ((0, 0),
                                      (padding_h, 1024 - shape(volume)[1] - padding_h),
                                      (0, 0)),
                             'constant')
        else:
            padding_x = int((512 - shape(volume)[2]) / 2)
            padding_h = 0
            volume = pad(volume, ((0, 0),
                                  (0, 0),
                                  (padding_x, 512 - shape(volume)[2] - padding_x)),
                         'constant')
    else:
        padding_h = 0
        padding_x = 0

    z_number = len(volume)
    start_slice = int((z_number-24)/2)+1
    end_slice = int((z_number-24)/2)+25
    volume = volume[start_slice:end_slice, :, :]
    volume = zoom(volume, (1, 0.5, 0.5))
    volume_torch = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(volume / 255), 0), 0)
    pred = model(volume_torch.to(device))
    pred_softmax = softmax(pred, dim=1)
    pred_mask = torch.argmax(pred_softmax, dim=1)
    pred_mask_cpu = pred_mask[0].cpu().data.numpy()

    p_c, p_m = common_mitotic_cell_regions_3d(pred_mask_cpu)
    p_c = morphology.remove_small_objects(array(p_c, dtype=bool), 1)  # 3.5
    p_m = morphology.remove_small_objects(array(p_m, dtype=bool), 6)  # 50
    region_c, _ = measure_region(p_c)
    region_m, _ = measure_region(p_m)
    common_regions = measure.regionprops(region_c)
    mitotic_regions = measure.regionprops(region_m)

    mitotic_centroids = []
    common_centroids = []
    for props in common_regions:
        z0, y0, x0 = props.centroid
        common_centroids.append([z0 + start_slice, (y0 * 2 - padding_h)*resize_parameter_h, (x0 * 2 -padding_x)*resize_parameter_w])

    for props in mitotic_regions:
        z0, y0, x0 = props.centroid
        mitotic_centroids.append([z0 + start_slice, (y0 * 2 - padding_h)*resize_parameter_h, (x0 * 2 - padding_x)*resize_parameter_w])
    return common_centroids, mitotic_centroids



