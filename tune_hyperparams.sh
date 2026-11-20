mkdir -p logs
for seed in 40 42 44; do
  for lr in 1e-4 5e-4 1e-3 5e-3; do
    for g in 0.9 0.933 0.95 0.98 0.99; do
      echo "$seed $lr $g"
    done
  done
done | xargs -P 12 -L 1 sh -c \
  'python3 -u main.py --seed $0 --models LSTM --lr $1 --gamma $2 --training_eps 50000 > logs/seed$0_LSTM_lr$1_gamma$2.txt 2>&1'