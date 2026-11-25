mkdir -p runs/smoke/logs
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param net --grad-method state --optimizer adam --seed 0 > runs/smoke/logs/P1_net_state_adam_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param net --grad-method state --optimizer lbfgs --seed 0 > runs/smoke/logs/P1_net_state_lbfgs_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param net --grad-method proxy --optimizer adam --seed 0 > runs/smoke/logs/P1_net_proxy_adam_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param net --grad-method proxy --optimizer lbfgs --seed 0 > runs/smoke/logs/P1_net_proxy_lbfgs_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param rates --grad-method state --optimizer lbfgs --seed 0 > runs/smoke/logs/P1_rates_state_lbfgs_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param net --grad-method state --optimizer lbfgs --train-without-pegs --seed 0 > runs/smoke/logs/P1_net_state_lbfgs_nopegs_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param endpose --window 20 --grad-method state --optimizer lbfgs --seed 0 > runs/smoke/logs/P1_endpose20_state_lbfgs_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param endpose --window 20 --aim --grad-method state --optimizer none --seed 0 > runs/smoke/logs/P1_endpose20aimed_state_none_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param endpose --window 60 --grad-method state --optimizer lbfgs --seed 0 > runs/smoke/logs/P1_endpose60_state_lbfgs_s0.log 2>&1
python scripts/train_shape.py train --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 --param endpose --window 60 --aim --grad-method state --optimizer none --seed 0 > runs/smoke/logs/P1_endpose60aimed_state_none_s0.log 2>&1
# after the runs above (greedy's step cap is the largest step of the state/L-BFGS solutions):
python scripts/train_shape.py greedy --config configs/task_shape_pegs_3d.yaml --out runs/smoke --target P1 > runs/smoke/logs/P1_greedy.log 2>&1
