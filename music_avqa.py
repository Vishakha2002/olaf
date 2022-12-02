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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set gpu number

import ast
import json
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import Dict, Tuple


class ToTensor(object):
    """
    XXX Need to understand why this is needed
    """

    def __call__(self, sample):

        audio = sample['audio']
        label = sample['label']

        return { 
                'audio': torch.from_numpy(audio), 
                'visual_posi': sample['visual_posi'], 
                'visual_nega': sample['visual_nega'],
                'question': sample['question'],
                'label': label}


class OlafInput(Dataset):

    def __init__(self, label, audio_vggish_features_dir, video_res14x14_dir, transform=None):
        super(Dataset, self).__init__()
        
        # We need to initialise it here once.
        self.word_to_idx = set()
        self.question_vocab = set()
        self.answer_vocab = set()
        # I believe this is what we would need to update in order to provide input from olaf frontend.
        # Vishakha what does label means here
        self.samples = self.load_json_data(label)
        # Max question length
        self.max_question_len = 14

        # why do we need this?
        # self.audio_vggish_features_dir = audio_vggish_features_dir
        # why do we need this?
        # self.video_res14x14_dir = video_res14x14_dir

        self.video_ids = {sample['video_id'] for sample in self.samples}
        self.total_sample_video_count = 60 * len(self.video_ids)
        self.transform = transform

        self.build_vocab()


        def __len__(self):
            return len(self.samples)


        def __getitem__(self, idx):

            sample = self.samples[idx]
            name = sample['video_id']
            # This is comming from vggish outout
            # audio = np.load(os.path.join(self.audio_dir, name + '.npy'))

        def make_sample(self):
            pass
            
                        
        def build_vocab(self) -> None:
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

            """
            question = []
            vocab_dataset_path = '/scratch/vtyagi14/data/json/avqa-train.json'
            # XXX Do we need a padding? 
            # question_vocab = ['<pad>']
            # question_vocab = set()


            vocab_dataset_dict = load_json_data(vocab_dataset_path)
            for sample in vocab_dataset_dict:
                question = sample['question_content'].rstrip().split(' ')
                question[-1] = question[-1][:-1]  # This is just removing the "?" from the last word

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
                    if '<' in question[index]:
                        question[index] = ast.literal_eval(sample['templ_values'])[found_object_index]
                        found_object_index += 1

            for word in question:
                self.question_vocab.add(word)
            
            self.word_to_idx = {word: i for i, word in enumerate(self.question_vocab)}
            
            # Since answers are only one word we didn't need to first convert it in a list.
            self.answer_vocab.add(sample['anser'])

        def load_json_data(filepath: str) -> Dict:
            """
            Load json and return dict to you
            """
            return json.load(open(filepath, 'r'))

if __name__ == "__main__":
    olaf_input_obj = OlafInput(
        label="/scratch/vtyagi14/data/json/avqa-test.json",
        audio_vggish_features_dir="/scratch/vtyagi14/data/feats/vggish",
        video_res14x14_dir="/scratch/vtyagi14/data/feats/res18_14x14",
        transform=transforms.Compose([ToTensor()]),
        )