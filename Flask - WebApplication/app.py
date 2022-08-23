import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from PIL import Image
from flask import Flask, flash, request, redirect, render_template, url_for
from flask import jsonify, send_file
from torch.utils.data import Dataset
from torchvision import transforms
from werkzeug.utils import secure_filename

# from app import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

data = []
model = []


class FashionDataset(Dataset):
    def __init__(self, json_data, root_dir, segments_dir, image_size, gen_img_sz):

        self.gen_img_sz = gen_img_sz
        self.meta_data = json_data
        self.root_dir = root_dir
        self.segments_dir = segments_dir
        self.transformGen = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(gen_img_sz, gen_img_sz)), torchvision.transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        self.transformDis = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(image_size, image_size)), torchvision.transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = list(self.meta_data[idx].keys())[0]
        # img_name = os.path.join(self.root_dir, filename)
        img_name = self.root_dir + '/' + filename
        image = Image.open(img_name).convert('RGB')

        gen_img = torch.randn(4, self.gen_img_sz, self.gen_img_sz)
        labels = []
        for img in self.meta_data[idx][filename]:
            # segImage = Image.open(os.path.join(self.segments_dir,img['segImage']))
            segImage = Image.open(self.segments_dir + '/' + img['segImage'])
            gen_img = gen_img + self.transformGen(segImage)
            labels.append(img['class'])

        sample = {'image': self.transformDis(image), 'inputImage': gen_img}

        return sample, self.get_lables(labels)

    def get_lables(self, labels):
        index = torch.tensor(labels)
        labels = torch.zeros(1, 13).index_fill_(1, index, 1)
        return labels.view(-1, 13)


class Generator(nn.Module):
    """ G(z) """

    def __init__(self, input_size=100):
        # initalize super module
        super(Generator, self).__init__()

        self.linear_y = nn.Sequential(nn.Linear(689, 507),
                                      nn.LeakyReLU(),
                                      )

        self.layer_xy = nn.Sequential(nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=5,
                                                         stride=2, padding=0, bias=False),
                                      nn.BatchNorm2d(2),
                                      nn.LeakyReLU(True),
                                      nn.Dropout(0.2),
                                      nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=5,
                                                         stride=2, padding=0, bias=False),
                                      nn.BatchNorm2d(2),
                                      nn.LeakyReLU(True),
                                      nn.Dropout(0.2),
                                      nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=5,
                                                         stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(2),
                                      nn.LeakyReLU(True),
                                      nn.Dropout(0.2),
                                      nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=4,
                                                         stride=1, padding=0, bias=False),
                                      nn.Tanh()
                                      )

    def forward(self, x, y):
        b_sz = x.shape[0]
        x = x.flatten()
        y = y.flatten()

        xy = torch.cat([x, y], dim=-1).view(b_sz, 1, -1)
        xy = self.linear_y(xy)

        xy = self.layer_xy(xy.view(b_sz, 3, 13, 13))
        return xy


@app.route('/')
def upload_form():
    return render_template('upload_final.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('upload_final.html', filenames=file_names)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/create_closet")
def create_closet():
    # saving images in memory
    closet_id = save_closet(request)
    # session['uploaded_img_file_path'] =
    return jsonify({"message": "Added Successfully", "closet_id": closet_id})


@app.route("/generate_styles", methods=['GET', 'POST'])
def generate_styles():
    if request.args['closet_id']:
        closet_id = int(request.args['closet_id'])
    print("Getting closet information")
    closet, lbl = get_closet(closet_id=closet_id)

    print("Generating style with model")
    fake_image = model(closet['inputImage'].view(1, 4, 13, 13), lbl.view(1, 1, 13)).cpu()
    img = vutils.make_grid(fake_image, padding=2, normalize=True)

    print("convert tensor to byte stream")
    img_byte = convert_to_bytestream(img, closet)

    # save
    cv2.imwrite('templates/123.jpg', img_byte)
    return send_file('templates/123.jpg', mimetype='image/jpg')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_closet(request):
    return random.randrange(10000, 20000, 50)


def load_model():
    try:
        print("Loading Model")
        global model
        model = Generator()
        model.load_state_dict(torch.load('netG.pt'))
        model.eval()
        print("Model loaded successfully")
    except Exception as ex:
        print("Error Loading model: ", ex)


def load_data():
    root_dir = "Data/Images"
    seg_dir = "Data/SegImages"

    f = open("Data/metadata_train.json")
    j_file = json.load(f)
    f.close()
    global data
    data = FashionDataset(j_file[:200], root_dir, seg_dir, 68, 13)


def get_closet(closet_id):
    i = np.random.randint(100, size=1)[0]
    images, lbl = data[i]

    return images, lbl


def convert_to_bytestream(img, closet):
    try:
        img_byte = np.transpose(img, (1, 2, 0)).numpy()
        img_byte = (img_byte * 255).astype(np.uint8)
        cv2.imwrite('123.jpg', img_byte)
        return img_byte
    except Exception as ex:
        print("Error convert_to_bytestream: ", ex)
        return None


def main():
    load_model()
    load_data()
    len(data)


if __name__ == "__main__":
    main()
    app.run(debug=True)
