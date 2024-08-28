import os
import torchaudio
import matplotlib.pyplot as plt
from torio._extension.utils import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

def plot_audio(filename):
    waveform, sample_rate = torchaudio.load(filename)
    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))
    
    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.title("Waveform")
    plt.show()
    
    return waveform, sample_rate

default_dir = os.getcwd()
folder = 'data'
print(f'Data directory will be: {default_dir}/{folder}')

if not os.path.isdir(folder):
    print("Creating folder.")
    os.mkdir(folder)
    torchaudio.datasets.SPEECHCOMMANDS(f'./{folder}/', download=True)
    os.chdir(f'./{folder}/SpeechCommands/speech_commands_v0.02/')
    labels = [name for name in os.listdir('.') if os.path.isdir(name)]
    # back to default directory
    os.chdir(default_dir)
    print(f'Total Labels: {len(labels)} \n')
    print(f'Label Names: {labels}')

# if os.path.isdir(folder):
#     filename = f"./{folder}/SpeechCommands/speech_commands_v0.02/yes/00f0204f_nohash_0.wav"
#     waveform, sample_rate = torchaudio.load(filename)
#     print(f'Waveform tensor:  {waveform}' )
#     waveform, sample_rate = plot_audio(filename)
#     ipd.Audio(waveform.numpy(), rate=sample_rate)

def load_audio_files(path: str, label: str):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))
    
    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        speaker_id, utterance_number = speaker.split('_nohash_')
        utterance_number = int(utterance_number)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append((waveform, sample_rate, label, speaker_id, utterance_number))
    
    return dataset

trainset_speechcommands_yes = load_audio_files(f'./{folder}/SpeechCommands/speech_commands_v0.02/yes', 'yes')
trainset_speechcommands_no = load_audio_files(f'./{folder}/SpeechCommands/speech_commands_v0.02/no', 'no')
print(f'Length of yes dataset: {len(trainset_speechcommands_yes)}')
print(f'Length of no dataset: {len(trainset_speechcommands_no)}')

trainloader_yes = torch.utils.data.DataLoader(trainset_speechcommands_yes, batch_size=1, shuffle=True)
trainloader_no = torch.utils.data.DataLoader(trainset_speechcommands_no, batch_size=1, shuffle=True)

yes_waveform = trainset_speechcommands_yes[0][0]
yes_sample_rate = trainset_speechcommands_yes[0][1]
# print(f'Yes Waveform: {yes_waveform}')
# print(f'Yes Sample Rate: {yes_sample_rate}')
# print(f'Yes Label: {trainset_speechcommands_yes[0][2]}')
# print(f'Yes ID: {trainset_speechcommands_yes[0][3]} \n')

no_waveform = trainset_speechcommands_no[0][0]
no_sample_rate = trainset_speechcommands_no[0][1]
# print(f'No Waveform: {no_waveform}')
# print(f'No Sample Rate: {no_sample_rate}')
# print(f'No Label: {trainset_speechcommands_no[0][2]}')
# print(f'No ID: {trainset_speechcommands_no[0][3]}')

def show_waveform(waveform, sample_rate, label):
    print("Waveform: {}\nSample rate: {}\nLabels: {} \n".format(waveform, sample_rate, label))
    new_sample_rate = sample_rate/10
    # Resample applies to a single channel, we resample first channel here
    channel = 0
    waveform_transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))
    print("Shape of transformed waveform: {}\nSample rate: {}".format(waveform_transformed.size(), new_sample_rate))
    plt.figure()
    plt.plot(waveform_transformed[0,:].numpy())
    plt.title("Waveform")
    plt.show()
    
# show_waveform(yes_waveform, yes_sample_rate, 'yes')
# show_waveform(no_waveform, no_sample_rate, 'no')

def show_spectrogram(waveform_classA, waveform_classB):
    yes_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classA)
    print("\nShape of yes spectrogram: {}".format(yes_spectrogram.size()))
    
    no_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classB)
    print("Shape of no spectrogram: {}".format(no_spectrogram.size()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Features of {}".format('no'))
    plt.imshow(yes_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
    plt.subplot(1, 2, 2)
    plt.title("Features of {}".format('yes'))
    plt.imshow(no_spectrogram.log2()[0,:,:].numpy(), cmap='viridis') 
    
    plt.show()

# show_spectrogram(yes_waveform, no_waveform)

def show_mfcc(waveform,sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0,:,:].numpy())
    plt.draw()
    plt.show()

# show_mfcc(yes_waveform, yes_sample_rate)

def create_spectrogram_images(trainloader, label_dir):
    #make directory
    directory = f'./{folder}/spectrograms/{label_dir}/'
    if(os.path.isdir(directory)):
        print("Data exists for", label_dir)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True)
        for i, data in enumerate(trainloader):
            waveform = data[0]
            sample_rate = data[1][0]
            label = data[2]
            ID = data[3]
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)     
            fig = plt.figure()
            plt.imsave(f'./data/spectrograms/{label_dir}/spec_img{i}.png', spectrogram_tensor[0].log2()[0,:,:].numpy(), cmap='viridis')
            plt.close(fig)

create_spectrogram_images(trainloader_yes, 'yes')
create_spectrogram_images(trainloader_no, 'no')