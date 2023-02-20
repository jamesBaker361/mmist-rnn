from models import *
from mnist_dijk import *
from evaluation import *
from data_processing import *
from datasets import load_dataset
import time
import argparse
import optuna


parser = argparse.ArgumentParser()
parser.add_argument("--path",type=str,help="path in gen imgs to save imgs",default="trial0")
parser.add_argument("--limit",type=int,default=2, help="how big dataset")
parser.add_argument('--labels', nargs='+', type=str, help='which labels (digits) to use')
parser.add_argument('--epochs', type=int, default=2, help="how many epochs to train")
parser.add_argument("--optuna", type=bool, default=False,help="whether to use optuna tuning")
parser.add_argument("--n_trials",type=int,default=5)
parser.add_argument("--lstm_embedding",type=int, default=4)
parser.add_argument("--lstm_units",type=int,default=256)
parser.add_argument("--lstm_layers",type=int,default=4)
parser.add_argument("--lstm_unit_multiplier",type=float,default=2.0)
parser.add_argument("--gpt_embedding",type=int,default=10)
parser.add_argument("--gpt_units",type=int,default=32)
parser.add_argument("--gpt_layers",type=int, default=3)
parser.add_argument("--gpt_unit_multiplier",type=float,default=1.0)
parser.add_argument("--gpt_heads",type=int,default=8)
parser.add_argument("--gpt_sequence_len",type=int,default=10)
parser.add_argument("--gpt_n_layers_class",type=int,default=2)
parser.add_argument("--gpt_units_class",type=int,default=16)
parser.add_argument("--gpt_activation_class", type=str, default="softmax")
parser.add_argument("--gpt_condition_stage", type=str, default="early")

parser.add_argument("--model", type=str,default="lstm",help='lstm or mini_gpt or mini_gpt_cond')
#parser.add_argument("--model")
args = parser.parse_args()

big_dataset = load_dataset("jlbaker361/mnist_sorted_v0.0",split="train")
train_dataset= big_dataset.filter(lambda example: example['split']=='train')
test_dataset=big_dataset.filter(lambda example: example['split']=='test')
balanced_dataset = get_balanced_dataset(train_dataset, args.limit, set(args.labels))
starting_points=get_starting_points(balanced_dataset)

def objective(trial):
    print(args)
    start=time.perf_counter()
    if args.model == 'lstm':
        inp_sequences, total_words = get_sequence_of_tokens(balanced_dataset['sequence'])
        predictors, label, max_sequence_len=generate_padded_sequences(inp_sequences)
        if args.optuna is True:
            args.lstm_embedding=trial.suggest_categorical("lstm_embedding",[2,4,8,16,32,64])
            args.lstm_units=trial.suggest_categorical("lstm_units",[16,32,64,128,256,512])
            args.lstm_layers=trial.suggest_int("lstm_layers",2,7)
            args.lstm_unit_multiplier=trial.suggest_categorical("lstm_unit_multiplier",[0.5,1.0,2])
        model = lstm_model(max_sequence_len-1,units=args.lstm_units,embedding_dim=args.lstm_embedding,n_layers=args.lstm_layers,unit_multiplier=args.lstm_unit_multiplier)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(predictors, label, epochs=args.epochs, verbose=2,callbacks=[GenImgCallback(starting_points,model, max_sequence_len, './gen_imgs/{}'.format(args.path))])
        result=model.evaluate(predictors,label)
    elif args.model == 'mini_gpt':
        if args.optuna is True:
            args.gpt_embedding=trial.suggest_categorical("gpt_embedding",[4,8,16,32,64])
            args.gpt_units=trial.suggest_categorical("gpt_units",[16,32,64,128,256,512])
            args.gpt_layers=trial.suggest_int("gpt_layers",2,7)
            args.gpt_unit_multiplier=trial.suggest_categorical("gpt_unit_multiplier",[0.5,1.0,2])
            args.gpt_heads=trial.suggest_categorical("gpt_heads",[4,8,16,32,64])
            args.gpt_sequence_len=trial.suggest_int("gpt_sequence_len",10,50)
            
        model= mini_gpt_model(
            args.gpt_sequence_len,
            units=args.gpt_units,
            embedding_dim=args.gpt_embedding,
            n_layers=args.gpt_layers,
            unit_multiplier=args.gpt_unit_multiplier,
            num_heads=args.gpt_heads)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        corpus=balanced_dataset["sequence"]
        x,y,subseq_labels= get_sequences_attention(args.gpt_sequence_len,corpus)
        test_balanced_dataset = get_balanced_dataset(test_dataset,args.limit, set(args.labels))
        test_corpus=test_balanced_dataset["sequence"]
        test_x,test_y, test_subseq_labels = get_sequences_attention(args.gpt_sequence_len,test_corpus)
        model.fit(x,y,epochs=args.epochs, callbacks=[GenImgCallbackAttention(starting_points,model, args.gpt_sequence_len, './gen_imgs/{}'.format(args.path),100)])
        result=model.evaluate(test_x,test_y)
    elif args.model== 'mini_gpt_cond' or args.model== 'mini_gpt_conditional':
        if args.optuna is True:
            args.gpt_embedding=trial.suggest_categorical("gpt_embedding",[4,8,16,32,64])
            args.gpt_units=trial.suggest_categorical("gpt_units",[16,32,64,128,256,512])
            args.gpt_layers=trial.suggest_int("gpt_layers",2,7)
            args.gpt_unit_multiplier=trial.suggest_categorical("gpt_unit_multiplier",[0.5,1.0,2])
            args.gpt_heads=trial.suggest_categorical("gpt_heads",[4,8,16,32,64])
            args.gpt_sequence_len=trial.suggest_int("gpt_sequence_len",10,50)
            args.gpt_n_layers_class=trial.suggest_int('gpt_n_layers_class',0,6)
            args.gpt_units_class=trial.suggest_categorical('gpt_units_class',[4,8,16,32])
            args.gpt_activation_class=trial.suggest_categorical('gpt_activation_class',['linear', 'relu','softmax','tanh'])
            args.gpt_condition_stage=trial.suggest_categorical('gpt_condition_stage',['both', 'early', 'late'])
        model=mini_gpt_model_conditional(
            args.gpt_sequence_len,
            units=args.gpt_units,
            embedding_dim=args.gpt_embedding,
            n_layers=args.gpt_layers,
            unit_multiplier=args.gpt_unit_multiplier,
            num_heads=args.gpt_heads,
            n_classes=len(args.labels),
            n_layers_class=args.n_layers_class,
            units_class=args.gpt_units_class,
            activation_class=args.gpt_activation_class,
            condition_stage=args.gpt_condition_stage
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        corpus=balanced_dataset["sequence"]
        labels=balanced_dataset["label"]
        x,y,subseq_labels= get_sequences_attention(args.gpt_sequence_len,corpus,labels=labels)
        expanded, oh_encoder=expand_labels(subseq_labels, max_seq_len=args.gpt_sequence_len)
        model.fit([x,expanded],y,epochs=args.epochs, callbacks=[GenImgCallbackAttentionConditional(oh_encoder,starting_points,model, args.gpt_sequence_len, './gen_imgs/{}'.format('args.path'),150)])
        
        test_balanced_dataset = get_balanced_dataset(test_dataset,args.limit, set(args.labels))
        test_corpus=test_balanced_dataset["sequence"]
        test_labels=test_balanced_dataset["label"]
        test_x,test_y, test_subseq_labels = get_sequences_attention(args.gpt_sequence_len,test_corpus, labels=test_labels)
        test_expanded, oh_encoder=expand_labels(test_subseq_labels, max_seq_len=args.gpt_sequence_len, oh_encoder=oh_encoder)
        
        result= model.evaluate([test_x,test_expanded],test_y)

    print('result = ',result)
    if args.optuna:
        return result

if __name__ == '__main__':
    if args.optuna:

        class CallBack:
            def __call__(self, study, trial) -> None:
                print("finished a trial!")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials, callbacks=[CallBack()], gc_after_trial=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        objective(None)
