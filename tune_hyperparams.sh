mkdir -p logs
for model in 'DisRNN' 'DisLRU'; do
  for lr in 1e-4 5e-4 1e-3 5e-3 1e-2; do
    for g in 0.9 0.95 0.98 0.995; do
      echo "$model $lr $g"
    done
  done
done | xargs -P 12 -L 1 sh -c \
  'python3 main.py --seed 40 --models $0 --lr $1 --gamma $2 --training_eps 100000 > logs/$0_lr$1_gamma$2.txt 2>&1'