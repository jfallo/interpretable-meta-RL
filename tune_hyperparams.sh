mkdir -p logs
for seed in 40 42 44; do
  for model in DisRNN DisLRU; do
    for g in 0.933 0.95; do
      echo "$seed $model $g"
    done
  done
done | xargs -P 12 -L 1 sh -c \
  'python3 -u main.py --seed $0 --models $1 --lr 5e-4 --gamma $2 --training_eps 200000 > logs/seed$0_$1_lr5e-4_gamma$2.txt 2>&1'