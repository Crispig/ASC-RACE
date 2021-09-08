import torch
import time
from multiprocessing import Pool
class RaceExample(object):

    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id = filename+'-'+str(i),
                            context_sentence = article,
                            start_ending = question,

                            ending_0 = options[0],
                            ending_1 = options[1],
                            ending_2 = options[2],
                            ending_3 = options[3],
                            label = truth))
                
    return examples 

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option 
    # 
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    max_option_len = 0
    examples_iter = tqdm(examples, desc="Preprocessing: ", disable=False) if is_main_process() else examples
    for example_index, example in enumerate(examples_iter):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        
        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label
            )
        )

    # torch.save(features,'./f.pt')

    return features

def transfer(args):
    return func(*args)

def func(load_path):
    k=torch.load(load_path)
    return k

def main():
    start=time.time()

    # x=torch.load('./f_320_orig.pt')
    arg_list=[]

    for i in range(36):
    #     if i!=35:
    #         torch.save(x[i*int(87866/36):(i+1)*int(87866/36)],'./pt_file/f{}.pt'.format(i))
    #     else:
    #         torch.save(x[i*int(87866/36):],'./pt_file/f{}.pt'.format(i))

        arg_list.append(['./pt_file/f{}.pt'.format(i)])

    pool=Pool(36)
    al=pool.map(transfer ,arg_list)

    l=[]
    for i in range(36):
        l=l+al[i]

    print(len(l))
    end=time.time()
    print("time",end-start)

if __name__ == "__main__":
    main()