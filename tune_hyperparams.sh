mkdir -p logs
for seed in 40 42 44; do
  for model in 'DisRNN' 'DisLRU'; do
    for lr in 1e-4 5e-4 1e-3 5e-3; do
      for g in 0.9 0.95 0.98 0.995; do
        echo "$seed $model $lr $g"
      done
    done
  done
done | xargs -P 12 -L 1 sh -c \
  'python3 -u main.py --seed $0 --models $1 --lr $2 --gamma $3 --training_eps 50000 > logs/seed$0_$1_lr$2_gamma$3.txt 2>&1'