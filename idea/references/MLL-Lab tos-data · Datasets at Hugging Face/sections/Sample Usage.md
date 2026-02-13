Sample Usage
To run a full pipeline evaluation (explore + eval + cogmap) using the provided scripts:

python scripts/SpatialGym/spatial_run.py \ --phase all \ --model-name gpt-5.2 \ --num 25 \ --data-dir room_data/3-room/ \ --output-root result/ \ --render-mode vision,text \ --exp-type active,passive \ --inference-mode batch