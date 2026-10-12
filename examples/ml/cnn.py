"""A VGG-style convolutional network drawn as 3D feature maps.

Block height and depth follow the spatial size, thickness follows the number
of channels; ``bands`` splits a block into stacked convolutions.
"""

import pytikz as tz

pic = tz.Picture()

cnn = pic.add(tz.ConvNet(spacing=0.7))
cnn.input(224, 3, label="input\nimage")
cnn.conv(64, 224, bands=2, label="conv1")
cnn.pool(112, label="pool")
cnn.conv(128, 112, bands=2, label="conv2")
cnn.pool(56, label="pool")
cnn.conv(256, 56, bands=3, label="conv3")
cnn.pool(28, label="pool")
cnn.conv(512, 28, bands=3, label="conv4")
cnn.pool(14, label="pool")
cnn.fc(4096, label="fc6")
cnn.fc(4096, label="fc7", gap=0.75)
cnn.fc(1000, label="softmax", color="softmax", gap=0.75)

# A brace over the convolutional part and one over the classifier.
x0, y0, x1, y1 = cnn.extent
features_end = cnn.block(8).extent[2]
pic.brace((cnn.block(1).extent[0], y1 + 0.25), (features_end, y1 + 0.25), "feature extraction")
pic.brace((cnn.block(9).extent[0], y1 + 0.25), (x1, y1 + 0.25), "classifier")

if __name__ == "__main__":
    pic.render("cnn.pdf")
