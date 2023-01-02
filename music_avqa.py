"""
Script to run arg MUSIC AVQA model using user inputs

https://blog.paperspace.com/dataloaders-abstractions-pytorch/
Start here is you need to understand Machine learning data loader class.

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


To run a test using MUSIC AVQA model:
python net_grd_avst/main_avst.py --mode test \
    --audio_dir = "path to your audio features"
    --video_res14x14_dir = "path to your visual res14x14 features"

test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                   transform=transforms.Compose([ToTensor()]), mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        test(model, test_loader)

Inputs:
    test_dataset:
        - label_val
        - label_test : "/scratch/vtyagi14/data/json/avqa-test.json"
        - audio_dir : /scratch/vtyagi14/data/feats/vggish
        - video_res14x14_dir : /scratch/vtyagi14/data/feats/res18_14x14
        - transform=transforms.Compose([ToTensor()])
        - mode_flag='test'

        self.ques_vocab <class 'list'> of lenght 93. Example <pad>  = Vishakha this need to remain as is.
        self.ans_vocab <class 'list'> of length 42. Example two  = Vishakha this need to remain as is.
        self.word_to_ix <class 'dict'> of 93. Example ('<pad>', 0)
        self.samples <class 'list'> of 9181. Example {'video_id': '00000093', 'question_id': 12, 'type': '["Audio", "Counting"]', 'question_content': 'How many musical instruments were heard throughout the video?', 'templ_values': '[]', 'question_deleted': 0, 'anser': 'one'}
        self.video_res14x14_dir /scratch/vtyagi14/data/feats/res18_14x14


Question format:

"""
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set gpu number

import ast
import json
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import Dict, Tuple


# class ToTensor(object):
#     """
#     XXX Need to understand why this is needed
#     """

#     def __call__(self, sample):

#         audio = sample['audio']
#         # label = sample['label']

#         return {
#                 'audio': torch.from_numpy(audio),
#                 'visual_posi': sample['visual_posi'],
#                 'visual_nega': sample['visual_nega'],
#                 'question': sample['question']}


class ToTensor(object):
    def __call__(self, sample):

        audio = sample["audio"]
        # visual_posi = sample['visual_posi']
        # visual_nega = sample['visual_nega']

        return {
            "audio": torch.from_numpy(audio),
            "visual_posi": sample["visual_posi"],
            "visual_nega": sample["visual_nega"],
            "question": sample["question"],
            "label": sample["label"],
        }


def load_json_data(filepath: str) -> Dict:
    """
    Load json and return dict to you
    """
    return json.load(open(filepath, "r"))


def ids_to_multinomial(id, categories):
    """label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]


class OlafInput(Dataset):
    def __init__(
        self,
        vocab_label,
        audio_vggish_features_dir,
        video_res14x14_dir,
        current_answer,
        transform=None,
        current_question=None,
        olaf_context={},
        is_batch=True,
    ):
        """
        Dataset class to represent data samples.
        is_batch: It is a flag to describe if we are doing batch processing or taking one input from Olaf
        vocab_label: Path to MUSIC_AVQA avqa-test.json
        """
        super(Dataset, self).__init__()

        # We need to initialise it here once.
        self.word_to_idx = set()
        self.question_vocab = set()
        self.answer_vocab = []
        self.is_batch = is_batch

        # I believe this is what we would need to update in order to provide input from olaf frontend.
        # XXX Vishakha what does label means here

        self.samples = load_json_data(vocab_label)
        # Max question length
        self.max_question_len = 14

        # XXX why do we need this?
        self.audio_vggish_features_dir = audio_vggish_features_dir
        # XXX why do we need this?
        self.video_res14x14_dir = video_res14x14_dir

        self.video_ids = {sample["video_id"] for sample in self.samples}
        self.total_sample_video_count = 60 * len(self.video_ids)
        self.transform = transform

        self.current_question = current_question
        self.olaf_context = olaf_context

        self.answer_label = {}

        self.build_vocab(vocab_label)
        # self.word_to_ix = list(self.word_to_idx)
        self.current_answer = current_answer
        # self.build_answer_label()

    def build_answer_label(self):
        for answer in self.answer_vocab:
            label = ids_to_multinomial(answer, self.answer_vocab)
            label = torch.from_numpy(np.array(label)).long()
            self.answer_label[label] = answer

    def build_vocab(self, vocab_label) -> None:
        """
        Read AVQA JSON training to build question and answer vocab for our model.
        Here we are using the same dataset provided to us by them.

        For now we will be using Training question answer dataset. Its a list of dict.
        Sample:
        {
            "video_id": "sa00007434",
            "question_id": 28884,
            "type": "[\"Visual\", \"Counting\"]",
            "question_content": "Are there <Object> and <Object> instruments in the video?",
            "templ_values": "[\"piano\", \"ukulele\"]",
            "question_deleted": 0,
            "anser": "no"
        }
        Path:
        /scratch/vtyagi14/data/json/avqa-train.json
        data/pretrained/avqa-train.json

        """
        question = []
        vocab_dataset_path = vocab_label
        # vocab_dataset_path = '/scratch/vtyagi14/data/json/avqa-train.json'
        # XXX Do we need a padding?
        # question_vocab = ['<pad>']
        # question_vocab = set()

        vocab_dataset_dict = load_json_data(vocab_dataset_path)
        for sample in vocab_dataset_dict:
            question = sample["question_content"].rstrip().split(" ")
            question[-1] = question[-1][
                :-1
            ]  # This is just removing the "?" from the last word

            # Now lets form the question by using templ_values
            # Sample Question:
            #     Are there <Object> and <Object> instruments in the video?
            # Sample Template values:
            #     "[\"piano\", \"ukulele\"]"
            # After running through literal_eval we will get the question string. Example:
            #     Are there piano and ukulele instruments in the video?
            found_object_index = 0
            for index in range(len(question)):
                # Looks for "<" in the word list. We are looking for word <Object>
                if "<" in question[index]:
                    question[index] = ast.literal_eval(sample["templ_values"])[
                        found_object_index
                    ]
                    found_object_index += 1

            for word in question:
                self.question_vocab.add(word)

            if sample["anser"] not in self.answer_vocab:
                self.answer_vocab.append(sample["anser"])

        current_question = self.current_question.split(" ")
        if current_question[-1] == "?":
            current_question[-1] = current_question[-1][:-1]
        for word in current_question:
            self.question_vocab.add(word)

        self.word_to_idx = {word: i for i, word in enumerate(self.question_vocab)}

    def __len__(self):
        if self.is_batch:
            return len(self.samples)

        return 1

    def __getitem__(self, idx):
        """
        sample = self.samples[idx]
        # sample example
        {
            "video_id": "00000093",
            "question_id": 12,
            "type": "[\"Audio\", \"Counting\"]",
            "question_content": "How many musical instruments were heard throughout the video?",
            "templ_values": "[]",
            "question_deleted": 0,
            "anser": "one"
        }
        # This is the name of the video. In our case, it would be the name of our video
        name = sample['video_id']
        # Load vggish audio features
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        # XXX What does it do? This seems like a 2 dimension array. and slicing is happening.
        # It seems we are selecting every 6th frame feature
        audio = audio[::6, :]

        # visual_out_res18_path = '/home/guangyao_li/dataset/avqa-features/visual_14x14'
        # Load resent18 features.
        visual_posi = np.load(os.path.join(self.video_res14x14_dir, name + '.npy'))

        # visual_posi [60, 512, 14, 14], select 10 frames from one video
        # XXX Why are we doing this??? It seems we are selecting every 6th frame feature
        visual_posi = visual_posi[::6, :]
        video_idx=self.video_list.index(name)

        for i in range(visual_posi.shape[0]):
            while(1):
                neg_frame_id = random.randint(0, self.video_len - 1)
                if (int(neg_frame_id/60) != video_idx):
                    break

            neg_video_id = int(neg_frame_id / 60)
            neg_frame_flag = neg_frame_id % 60
            neg_video_name = self.video_list[neg_video_id]
            visual_nega_out_res18=np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'))

            visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)
            visual_nega_clip=visual_nega_out_res18[neg_frame_flag,:,:,:].unsqueeze(0)

            if(i==0):
                visual_nega=visual_nega_clip
            else:
                visual_nega=torch.cat((visual_nega,visual_nega_clip),dim=0)

        # visual nega [60, 512, 14, 14]

        # question
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        sample = {'audio': audio, 'visual_posi': visual_posi, 'visual_nega': visual_nega, 'question': ques, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        {
            "video_title": "Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video",
            "raw_audio": "/home/vishakha/olaf/data/raw_audio/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.wav",
            "raw_video": "/home/vishakha/olaf/data/raw_video/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.mp4",
            "video_frame_path": "/home/vishakha/olaf/data/frames/video/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video",
            "vggish_audio_feature_file_path": "/home/vishakha/olaf/data/features/audio_vggish/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.npy",
            "resnet_video_feature_file_path": "/home/vishakha/olaf/data/features/video_resnet18/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.npy",
            "extracted_frames": True
        }

        return sample
        """
        # XXX Comeup with a better name and reset default as you become aware of them
        # single_sample = {
        #     'audio': None,
        #     'visual_posi': None,
        #     'visual_nega': None,
        #     'question': None,
        #     'label': None
        # }

        if not self.is_batch:
            # If we are here that means we need to take input from olaf rather then do batch processing.

            vggish_audio_feature = np.load(
                self.olaf_context.get("vggish_audio_feature_file_path")
            )
            print("Vishakha vggish_audio_feature before [::6, :] ~~~~~")
            print(f"{vggish_audio_feature.shape}")
            # print("Vishakha vggish_audio_feature here ~~~~~")
            # print(f"{vggish_audio_feature.shape}")
            # XXX TODO What does it do?
            # This seems like a 2 dimension array. and slicing is happening.
            # It seems we are selecting every 6th frame feature
            vggish_audio_feature = vggish_audio_feature[::6, :]
            print("Vishakha vggish_audio_feature after [::6, :] ~~~~~")
            print(f"{vggish_audio_feature.shape}")

            # resnet_video_feature
            visual_posi = np.load(
                self.olaf_context.get("resnet_video_feature_file_path")
            )
            print("Vishakha visual_posi before [::6, :] ~~~~~")
            print(f"{visual_posi.shape}")
            # XXX Why are we doing this??? It seems we are selecting every 6th frame feature
            visual_posi = visual_posi[::6, :]
            print("Vishakha visual_posi after [::6, :] ~~~~~")
            print(f"{visual_posi.shape}")
            visual_posi = visual_posi[:-1, :]
            print("Vishakha visual_posi after [:-1, :] ~~~~~")
            print(f"{visual_posi.shape}")

            # I am not sure what is happening here but trying to reverse engineer their code.
            # SOT https://github.com/GeWu-Lab/MUSIC-AVQA/blob/10420dce9df1e27e82500da31c18efdba98bc077/net_grd_avst/dataloader_avst.py#L135
            for i in range(visual_posi.shape[0]):
                neg_frame_id = random.randint(0, self.total_sample_video_count - 1)

                neg_video_id = int(neg_frame_id / 60)
                neg_frame_flag = neg_frame_id % 60
                neg_video_name = list(self.video_ids)[neg_video_id]

                visual_nega_out_res18 = np.load(
                    os.path.join(self.video_res14x14_dir, neg_video_name + ".npy")
                )

                visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)
                visual_nega_clip = visual_nega_out_res18[
                    neg_frame_flag, :, :, :
                ].unsqueeze(0)

                if i == 0:
                    visual_nega = visual_nega_clip
                else:
                    visual_nega = torch.cat((visual_nega, visual_nega_clip), dim=0)

            print(f"vishakha visual_nega shape {visual_nega.shape}")

            question = self.current_question.split(" ")
            if question[-1] == "?":
                question[-1] = question[-1][:-1]

            idxs = [self.word_to_idx[w] for w in question]
            ques = torch.tensor(idxs, dtype=torch.long)

            label = ids_to_multinomial(self.current_answer, self.answer_vocab)
            label = torch.from_numpy(np.array(label)).long()

            single_sample = {
                "audio": vggish_audio_feature,
                "visual_posi": visual_posi,
                "visual_nega": visual_nega,
                "question": ques,
                "label": label,
            }

            if self.transform:
                sample = self.transform(single_sample)
            return sample

        raise Exception("Not Implemented for batch processing yet")
        # XXX Vishakha this code is almost implement at the following link.
        # https://github.com/GeWu-Lab/MUSIC-AVQA/blob/main/net_grd_avst/dataloader_avst.py#L121


class OlafBatchInput(Dataset):
    def __init__(
        self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag="train"
    ):
        samples = json.load(open("./data/pretrained/avqa-train.json", "r"))

        # nax =  nne
        ques_vocab = ["<pad>"]
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample["question_content"].rstrip().split(" ")
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if "<" in question[pos]:
                    question[pos] = ast.literal_eval(sample["templ_values"])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample["anser"] not in ans_vocab:
                ans_vocab.append(sample["anser"])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, "r"))
        self.max_len = 14  # question length

        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_res14x14_dir
        self.transform = transform

        video_list = []
        for sample in self.samples:
            video_name = sample["video_id"]
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.video_len = 60 * len(video_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        name = sample["video_id"]
        audio = np.load(os.path.join(self.audio_dir, name + ".npy"))
        audio = audio[::6, :]

        # visual_out_res18_path = '/home/guangyao_li/dataset/avqa-features/visual_14x14'
        visual_posi = np.load(os.path.join(self.video_res14x14_dir, name + ".npy"))

        # visual_posi [60, 512, 14, 14], select 10 frames from one video
        visual_posi = visual_posi[::6, :]

        video_idx = self.video_list.index(name)

        for i in range(visual_posi.shape[0]):
            while 1:
                neg_frame_id = random.randint(0, self.video_len - 1)
                if int(neg_frame_id / 60) != video_idx:
                    break

            neg_video_id = int(neg_frame_id / 60)
            neg_frame_flag = neg_frame_id % 60
            neg_video_name = self.video_list[neg_video_id]
            visual_nega_out_res18 = np.load(
                os.path.join(self.video_res14x14_dir, neg_video_name + ".npy")
            )

            visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)
            if visual_nega_out_res18.shape[0] < neg_frame_flag:
                print(f"Vishakha visual_nega_out_res18 shape {visual_nega_out_res18.shape[0]} and neg_frame_flag {neg_frame_flag}")
                neg_frame_flag = neg_frame_flag - 10
                print(f"Vishakha after neg_frame_flag {neg_frame_flag}")
            if neg_frame_flag == 50:
                print(f"Vishakha before neg_frame_flag {neg_frame_flag}")
                neg_frame_flag = random.randint(5,49)
                print(f"Vishakha after neg_frame_flag {neg_frame_flag}")

            visual_nega_clip = visual_nega_out_res18[neg_frame_flag, :, :, :].unsqueeze(
                0
            )

            if i == 0:
                visual_nega = visual_nega_clip
            else:
                visual_nega = torch.cat((visual_nega, visual_nega_clip), dim=0)

        # visual nega [60, 512, 14, 14]

        # question
        question_id = sample["question_id"]
        question = sample["question_content"].rstrip().split(" ")
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if "<" in question[pos]:
                question[pos] = ast.literal_eval(sample["templ_values"])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append("<pad>")
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample["anser"]
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        sample = {
            "audio": audio,
            "visual_posi": visual_posi,
            "visual_nega": visual_nega,
            "question": ques,
            "label": label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open("./data/pretrained/olaf-test.json", "r"))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, visual_posi, visual_nega, target, question = (
                sample["audio"].to("cuda"),
                sample["visual_posi"].to("cuda"),
                sample["visual_nega"].to("cuda"),
                sample["label"].to("cuda"),
                sample["question"].to("cuda"),
            )

            preds_qa, out_match_posi, out_match_nega = model(
                audio, visual_posi, visual_nega, question
            )
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type = ast.literal_eval(x["type"])
            if type[0] == "Audio":
                if type[1] == "Counting":
                    A_count.append((predicted == target).sum().item())
                elif type[1] == "Comparative":
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == "Visual":
                if type[1] == "Counting":
                    V_count.append((predicted == target).sum().item())
                elif type[1] == "Location":
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == "Audio-Visual":
                if type[1] == "Existential":
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == "Counting":
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == "Location":
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == "Comparative":
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == "Temporal":
                    AV_temp.append((predicted == target).sum().item())

    print("Audio Counting Accuracy: %.2f %%" % (100 * sum(A_count) / len(A_count)))
    if len(A_cmp) != 0:
        print("Audio Cmp Accuracy: %.2f %%" % (100 * sum(A_cmp) / len(A_cmp)))
        print(
            "Audio Accuracy: %.2f %%"
            % (100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp)))
        )
    if len(V_count) != 0:
        print("Visual Counting Accuracy: %.2f %%" % (100 * sum(V_count) / len(V_count)))
    if len(V_loc) != 0:
        print("Visual Loc Accuracy: %.2f %%" % (100 * sum(V_loc) / len(V_loc)))
        print(
            "Visual Accuracy: %.2f %%"
            % (100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc)))
        )
    if len(AV_ext) != 0:
        print("AV Ext Accuracy: %.2f %%" % (100 * sum(AV_ext) / len(AV_ext)))
    if len(AV_count) != 0:
        print("AV counting Accuracy: %.2f %%" % (100 * sum(AV_count) / len(AV_count)))
    if len(AV_loc) != 0:
        print("AV Loc Accuracy: %.2f %%" % (100 * sum(AV_loc) / len(AV_loc)))
    if len(AV_cmp) != 0:
        print("AV Cmp Accuracy: %.2f %%" % (100 * sum(AV_cmp) / len(AV_cmp)))
    if len(AV_temp) != 0:
        print("AV Temporal Accuracy: %.2f %%" % (100 * sum(AV_temp) / len(AV_temp)))
    if len(AV_cmp) != 0 and len(AV_temp) != 0:
        print(
            "AV Accuracy: %.2f %%"
            % (
                100
                * (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp) + sum(AV_cmp))
                / (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))
            )
        )

    print("Overall Accuracy: %.2f %%" % (100 * correct / total))

    return 100 * correct / total


if __name__ == "__main__":
    # This is from Lab Server
    # olaf_context = {
    #     "video_title": "Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video",
    #     "raw_audio": "/home/vishakha/olaf/data/raw_audio/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.wav",
    #     "raw_video": "/home/vishakha/olaf/data/raw_video/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.mp4",
    #     "video_frame_path": "/home/vishakha/olaf/data/frames/video/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video",
    #     "vggish_audio_feature_file_path": "/home/vishakha/olaf/data/features/audio_vggish/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.npy",
    #     "resnet_video_feature_file_path": "/home/vishakha/olaf/data/features/video_resnet18/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.npy",
    #     "extracted_frames": True
    # }
    # This is from Suoer computer Server
    olaf_context = {
        "video_title": "Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video",
        "raw_audio": "/home/vtyagi14/olaf/data/raw_audio/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.wav",
        "raw_video": "/home/vtyagi14/olaf/data/raw_video/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.mp4",
        "video_frame_path": "/home/vtyagi14/olaf/data/frames/video/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video",
        "vggish_audio_feature_file_path": "/home/vtyagi14/olaf/data/features/audio_vggish/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.npy",
        "resnet_video_feature_file_path": "/home/vtyagi14/olaf/data/features/video_resnet18/Marcin__Moonlight_Sonata_on_One_Guitar_Official_Video.npy",
        "extracted_frames": True,
    }

    olaf_input_obj = OlafInput(
        vocab_label="data/pretrained/avqa-train.json",
        audio_vggish_features_dir="/scratch/vtyagi14/data/feats/vggish",
        video_res14x14_dir="/scratch/vtyagi14/data/feats/res18_14x14",
        current_answer="one",
        transform=transforms.Compose([ToTensor()]),
        current_question="How many instruments are sounding in the video?",
        olaf_context=olaf_context,
        is_batch=False,
    )
    olaf_input_obj.__getitem__(0)
