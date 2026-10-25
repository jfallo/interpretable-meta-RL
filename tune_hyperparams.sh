mkdir -p logs
for seed in 40 42 44; do
  for model in DisRNN DisLRU; do
    for lrpair in 5e-4,0.0005 1e-3,0.001; do
      lrlog=${lrpair%,*}; lrck=${lrpair#*,}
      for g in 0.9 0.933 0.95; do
        echo "$seed $model $lrlog $lrck $g"
      done
    done
  done
done | xargs -P 12 -L 1 sh -c \
  'python3 -u main.py --seed $0 --models $1 --lr $3 --gamma $4 --training_eps 100000 \
    --resume checkpoints/bandits/independent/standard/$1/lr$3_gamma$4/seed$0/checkpoint_ep49999.pt \
    >> logs/seed$0_$1_lr$2_gamma$4.txt 2>&1'