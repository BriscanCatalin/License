# This is a sample Python script.

from colorsys import rgb_to_hsv
from tkinter import *
from tkinter import filedialog

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torchvision import models
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import transforms

filepath = ""
filepath_copy = ""
feature_extract = True
num_classes = 2
input_size = 224
ckp_path = "./"
in_width = 224
in_height = 224

data_transform = transforms.Compose([
    transforms.Resize((in_width, in_height)),
    transforms.ToTensor()
])

data_transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def openFileForPath():
    filepath = filedialog.askopenfilename(initialdir="",
                                          title="Open file okay?",
                                          filetypes=(("image files", "*.jpeg"),
                                                     ("png files", "*.png"),
                                                     ("jpg files", "*.jpg"),
                                                     ("all files", "*.*")))
    return filepath


def load_ckp(checkpoint_fpath, model, inception=False):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    print(checkpoint_fpath)

    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint, strict=False)

    return model


def displayResults(model):
    print("model = ", model)


# browse image and display it
def browseHandler():
    global filepath
    global filepath_copy
    filepath = openFileForPath()
    filepath_copy = filepath
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    scale_percent = 220  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    if 'pneumonia' in filepath_copy:
        if width > 1000:
            width = int(img.shape[1] * scale_percent / 100) % 1500
        if height > 1000:
            height = int(img.shape[0] * scale_percent / 100) % 1500
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    dim = (width, height)

    FONT_SCALE = 2e-3
    font_scale = min(width, height) * FONT_SCALE
    cv2.putText(img, 'Press any key', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale/2,
                (25, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'to close the image ... ', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale / 2,
                (25, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Image Window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(filepath_copy)


def setup_gui(model_ft, inception=False):
    test_img = Image.open(filepath_copy).convert("L")
    result = 0
    color_image = ImageOps.colorize(test_img,
                                    (0, 0, 0),
                                    (255, 255, 255))
    if inception:
        transformed_img = data_transform_inception(color_image)
    else:
        transformed_img = data_transform(color_image)
    input_img = transform_normalize(transformed_img)

    # the model requires a dummy batch dimension
    input_img = input_img.unsqueeze(0)

    model_ft.eval()

    # get predicted browning level
    output = model_ft(input_img)
    output = F.softmax(output, dim=1)

    # get first 2 maxima of output
    psv, plv = torch.topk(output, 2)
    plv.squeeze_()

    confidence = '%.6f' % psv.squeeze()[0].squeeze().item()
    img = cv2.imread(filepath_copy)
    predicted_label = plv[0].item()

    if predicted_label == 1:
        if 'brain' in filepath_copy:
            prediction = 'Brain Tumor'
        elif 'pneumonia' in filepath_copy:
            prediction = 'Pneumonia'
    else:
        prediction = 'Normal'

    scale_percent = 220  # percent of original size
    width = int(img.shape[1] * scale_percent / 150) % 1500
    height = int(img.shape[0] * scale_percent / 150) % 1500
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    FONT_SCALE = 2e-3
    font_scale = min(width, height) * FONT_SCALE

    cv2.putText(resized, 'Confidence:' + confidence, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (25, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(resized, 'Label:' + prediction, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (25, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def inceptionHandlerBrain():
    print("Inception Brain")
    """ Inception v3 Be careful, expects (299,299) sized images and has auxiliary output  """
    model_ft = models.inception_v3(init_weights=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierInceptionBrain.pt",
        model_ft, True)
    setup_gui(model, True)
    displayResults(model)


def resnetHandlerBrain():
    print("Resnet Brain")
    model_ft = models.resnet18()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierResnetBrain.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def denseHandlerBrain():
    print("Dense Brain")
    model_ft = models.densenet121()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierDenseBrain.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def vggHandlerBrain():
    print("Vgg Brain")
    model_ft = models.vgg11_bn()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierVggBrainModified2.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def alexHandlerBrain():
    print("Alexnet Brain")
    model_ft = models.alexnet()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    model = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierAlexBrain.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def squeezeHandlerBrain():
    print("Squeeze Brain")
    model_ft = models.squeezenet1_0()
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes
    model_ft = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierSqueezeBrain.pt",
        model_ft)
    setup_gui(model_ft)
    displayResults(model_ft)


def denseHandlerBrainModified():
    print("Dense Brain FineTuned")
    model_ft = models.densenet121()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model_ft.classifier = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=265),
        nn.ReLU(),
        nn.Linear(in_features=265, out_features=2)
    )
    model_ft = load_ckp(
        "../License Models/SavedModelsSimpleBrain/classifierDenseBrainModified.pt",
        model_ft)
    setup_gui(model_ft)
    displayResults(model_ft)


def inceptionHandlerPneumonia():
    print("Inception Pneumonia")
    model_ft = models.inception_v3()
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierInceptionPneumonia.pt",
        model_ft, True)
    setup_gui(model, True)
    displayResults(model)


def resnetHandlerPneumonia():
    print("Resnet Pneumonia")
    model_ft = models.resnet18()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierResnetPneumonia.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def denseHandlerPneumonia():
    print("Dense Pneumonia")
    model_ft = models.densenet121()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierDensePneumonia.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def vggHandlerPneumonia():
    print("Vgg Pneumonia")
    model_ft = models.vgg11_bn(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierVggPneumoModified2.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def alexHandlerPneumonia():
    print("Alexnet Pneumonia")
    model_ft = models.alexnet()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    model = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierAlexPneumonia.pt",
        model_ft)
    setup_gui(model)
    displayResults(model)


def squeezeHandlerPneumonia():
    print("Squeeze Pneumonia")
    model_ft = models.squeezenet1_0()
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes
    model_ft = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierSqueezePneumonia.pt",
        model_ft)
    setup_gui(model_ft)
    displayResults(model_ft)


def denseHandlerPneumoniaFinetuned():
    print("Dense Pneumonia FineTuned")
    model_ft = models.densenet121()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model_ft.classifier = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=265),
        nn.ReLU(),
        nn.Linear(in_features=265, out_features=2)
    )
    model_ft = load_ckp(
        "../License Models/SavedModelsSimplePneumonia/classifierDensePneumoniaModified.pt",
        model_ft)
    setup_gui(model_ft)
    displayResults(model_ft)


def effB0HandlerBrain():
    print("Efficient B0 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb0Brain.pt",
        model_ft, )

    setup_gui(model)
    displayResults(model)


def effB1HandlerBrain():
    print("Efficient B1 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b1')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb1Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB2HandlerBrain():
    print("Efficient B2 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b2')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb2Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB3HandlerBrain():
    print("Efficient B3 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b3')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb3Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB4HandlerBrain():
    print("Efficient B4 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b4')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb4Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB5HandlerBrain():
    print("Efficient B5 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b5')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb5Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB6HandlerBrain():
    print("Efficient B6 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b6')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb6Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB7HandlerBrain():
    print("Efficient B7 Brain")
    model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb7Brain.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB0HandlerBrainFineTuned():
    print("Efficient B0 Brain FineTuned")
    model_ft = models.efficientnet_b0(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(in_features=1280, out_features=625),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(in_features=625, out_features=256),
                                        nn.ReLU(),
                                        nn.Linear(in_features=256, out_features=num_classes)
                                        )
    model = load_ckp(
        "../License Models/SavedModelsEfficientBrain/classifierEffb0BrainModified.pt",
        model_ft, )

    setup_gui(model)
    displayResults(model)


def effB0HandlerPneumonia():
    print("Efficient B0 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb0Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB1HandlerPneumonia():
    print("Efficient B1 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b1')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb1Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB2HandlerPneumonia():
    print("Efficient B2 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b2')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb2Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB3HandlerPneumonia():
    print("Efficient B3 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b3')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb3Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB4HandlerPneumonia():
    print("Efficient B4 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b4')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb4Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB5HandlerPneumonia():
    print("Efficient B5 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b5')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb5Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB6HandlerPneumonia():
    print("Efficient B6 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b6')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb6Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB7HandlerPneumonia():
    print("Efficient B7 Pneumonia")
    model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb7Pneumonia.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def effB0HandlerPneumoniaFineTuned():
    print("Efficient B0 Pneumonia FineTunde")
    model_ft = models.efficientnet_b0(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    # modify the network to accomodate num_classes
    model_ft.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(in_features=1280, out_features=625),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(in_features=625, out_features=256),
                                        nn.ReLU(),
                                        nn.Linear(in_features=256, out_features=num_classes)
                                        )
    model = load_ckp(
        "../License Models/SavedModelsEfficientPneumonia/classifierEffb0PneumoniaModified.pt",
        model_ft)

    setup_gui(model)
    displayResults(model)


def rgb_hack(rgb):
    return "#%02x%02x%02x" % rgb


# creates a Tk() object
master = Tk(className="Pneumo-Brainly")

# sets the geometry of main
# root window
master.geometry("2560x1440")

C = Canvas(master, height=50)
filename = PhotoImage(file="images.png")
background_label = Label(master, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
C.pack()

label = Button(master, text="Welcome to Pneumo-Brainly!", width=40, highlightbackground="#2f3542")
label.pack(pady=100, padx=100, side=TOP)
label.place(x=140, y=70)

top = Button(master, text="Browse an image ...", width=20, highlightbackground="#2f3541",
             bg="black", command=browseHandler)
top.pack(pady=20)
top.place(x=140, y=120)

panel_up = PanedWindow(master, orient=VERTICAL)
panel_left = PanedWindow(master, orient=VERTICAL, width=200)
panel_left.pack(side=LEFT)
panel_left.configure(bg=rgb_hack((27, 29, 40)))
panel_up.pack(side=LEFT, expand=True)
panel_up.add(panel_left)

panel_right = PanedWindow(master, orient=VERTICAL)
panel_right.pack(side=RIGHT)
panel_right.configure(bg=rgb_hack((27, 29, 40)))
panel_up.add(panel_right)
panel_up.configure(width=500, bg=rgb_hack((27, 29, 40)))

panel_down = PanedWindow(master, orient=VERTICAL, width=200)
panel_right_right = PanedWindow(master, orient=VERTICAL)
panel_right_right.pack(side=RIGHT)
panel_right_right.configure(bg=rgb_hack((27, 29, 40)))
panel_down.pack(side=RIGHT, expand=True)
panel_down.add(panel_right_right)

panel_left_left = PanedWindow(master, orient=VERTICAL)
panel_left_left.pack(side=LEFT)
panel_left_left.configure(bg=rgb_hack((27, 29, 40)))
panel_down.add(panel_left_left)

panel_down.configure(width=500)

# Handlers for simple brain modelsfg='red',
btnInception = Button(master, text="Inception Handler Brain",
                      highlightbackground="#2f3541",
                      command=inceptionHandlerBrain)
btnInception.pack(pady=50)
panel_left.add(btnInception)
btnDense = Button(master, text="Dense Handler Brain",
                  highlightbackground="#2f3541",
                  command=denseHandlerBrain)
btnDense.pack(pady=50)
panel_left.add(btnDense)
btnVgg = Button(master, text="Vgg Handler Brain",
                highlightbackground="#2f3541",
                command=vggHandlerBrain)
btnVgg.pack(pady=50)
panel_left.add(btnVgg)
btnAlex = Button(master, text="Alex Handler Brain",
                 highlightbackground="#2f3541",
                 command=alexHandlerBrain)
btnAlex.pack(pady=50)
panel_left.add(btnAlex)
btnResnet = Button(master, text="Resnet Handler Brain",
                   highlightbackground="#2f3541",
                   command=resnetHandlerBrain)
btnResnet.pack(pady=50)
panel_left.add(btnResnet)
btnSqueeze = Button(master, text="Squeeze Handler Brain",
                    highlightbackground="#2f3541",
                    command=squeezeHandlerBrain)
btnSqueeze.pack(pady=50)
panel_left.add(btnSqueeze)

btnDenseModified = Button(master, text="Dense Handler Brain FineTuned",
                          highlightbackground="#2f3541",
                          command=denseHandlerBrainModified)
btnDenseModified.pack(pady=50)
panel_left.add(btnDenseModified)

# Handlers for simple pneumonia models
btnInceptionP = Button(master, text="Inception Handler Pneumonia", command=inceptionHandlerPneumonia)
btnInceptionP.pack(pady=50)
panel_left_left.add(btnInceptionP)
btnResnetP = Button(master, text="Resnet Handler Pneumonia", command=resnetHandlerPneumonia)
btnResnetP.pack(pady=50)
panel_left_left.add(btnResnetP)
btnDenseP = Button(master, text="Dense Handler Pneumonia", command=denseHandlerPneumonia)
btnDenseP.pack(pady=50)
panel_left_left.add(btnDenseP)
btnVggP = Button(master, text="Vgg Handler Pneumonia", command=vggHandlerPneumonia)
btnVggP.pack(pady=50)
panel_left_left.add(btnVggP)
btnAlexP = Button(master, text="Alex Handler Pneumonia", command=alexHandlerPneumonia)
btnAlexP.pack(pady=50)
panel_left_left.add(btnAlexP)
btnSqueezeP = Button(master, text="Squeeze Handler Pneumonia", command=squeezeHandlerPneumonia)
btnSqueezeP.pack(pady=50)
panel_left_left.add(btnSqueezeP)

btnDensePModified = Button(master, text="Dense Handler Pneumonia FineTunded", command=denseHandlerPneumoniaFinetuned)
btnDensePModified.pack(pady=50)
panel_left_left.add(btnDensePModified)

# Handlers for efficient brain models
btnEffB0 = Button(master, text="EffB0 Handler Brain",
                  highlightbackground="#2f3541", command=effB0HandlerBrain)
btnEffB0.pack(pady=50)
panel_right.add(btnEffB0)
btnEffB1 = Button(master, text="EffB1 Handler Brain",
                  highlightbackground="#2f3541", command=effB1HandlerBrain)
btnEffB1.pack(pady=50)
panel_right.add(btnEffB1)
btnEffB2 = Button(master, text="EffB2 Handler Brain",
                  highlightbackground="#2f3541", command=effB2HandlerBrain)
btnEffB2.pack(pady=50)
panel_right.add(btnEffB2)
btnEffB3 = Button(master, text="EffB3 Handler Brain",
                  highlightbackground="#2f3541", command=effB3HandlerBrain)
btnEffB3.pack(pady=50)
panel_right.add(btnEffB3)
btnEffB4 = Button(master, text="EffB4 Handler Brain",
                  highlightbackground="#2f3541", command=effB4HandlerBrain)
btnEffB4.pack(pady=50)
panel_right.add(btnEffB4)
btnEffB5 = Button(master, text="EffB5 Handler Brain",
                  highlightbackground="#2f3541", command=effB5HandlerBrain)
btnEffB5.pack(pady=50)
panel_right.add(btnEffB5)
btnEffB6 = Button(master, text="EffB6 Handler Brain",
                  highlightbackground="#2f3541", command=effB6HandlerBrain)
btnEffB6.pack(pady=50)
panel_right.add(btnEffB6)
btnEffB7 = Button(master, text="EffB7 Handler Brain",
                  highlightbackground="#2f3541", command=effB7HandlerBrain)
btnEffB7.pack(pady=50)
panel_right.add(btnEffB7)
btnEffB0Modified = Button(master, text="EffB0 Handler Brain FineTuned",
                          highlightbackground="#2f3541", command=effB0HandlerBrainFineTuned)
btnEffB0Modified.pack(pady=50)
panel_right.add(btnEffB0Modified)
# Handlers for efficient pneumonia models
btnEffB0P = Button(master, text="EffB0 Handler Pneumonia", command=effB0HandlerPneumonia)
btnEffB0P.pack(pady=50)
panel_right_right.add(btnEffB0P)
btnEffB1P = Button(master, text="EffB1 Handler Pneumonia", command=effB1HandlerPneumonia)
btnEffB1P.pack(pady=50)
panel_right_right.add(btnEffB1P)
btnEffB2P = Button(master, text="EffB2 Handler Pneumonia", command=effB2HandlerPneumonia)
btnEffB2P.pack(pady=50)
panel_right_right.add(btnEffB2P)
btnEffB3P = Button(master, text="EffB3 Handler Pneumonia", command=effB3HandlerPneumonia)
btnEffB3P.pack(pady=50)
panel_right_right.add(btnEffB3P)
btnEffB4P = Button(master, text="EffB4 Handler Pneumonia", command=effB4HandlerPneumonia)
btnEffB4P.pack(pady=50)
panel_right_right.add(btnEffB4P)
btnEffB5P = Button(master, text="EffB5 Handler Pneumonia", command=effB5HandlerPneumonia)
btnEffB5P.pack(pady=50)
panel_right_right.add(btnEffB5P)
btnEffB6P = Button(master, text="EffB6 Handler Pneumonia", command=effB6HandlerPneumonia)
btnEffB6P.pack(pady=50)
panel_right_right.add(btnEffB6P)
btnEffB7P = Button(master, text="EffB7 Handler Pneumonia", command=effB7HandlerPneumonia)
btnEffB7P.pack(pady=50)
panel_right_right.add(btnEffB7P)
btnEffB0ModifiedP = Button(master, text="EffB0 Handler Pneumonia FineTuned", command=effB0HandlerPneumoniaFineTuned)
btnEffB0ModifiedP.pack(pady=50)
panel_right_right.add(btnEffB0ModifiedP)

mainloop()
