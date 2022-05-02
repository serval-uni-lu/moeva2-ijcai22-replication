# RQ1
python -m src.run_rq1 -c ./config/rq1.lcld.yaml
python -m src.run_rq1 -c ./config/rq1.botnet.yaml
#
## RQ2
python -m src.run_rq2 -c ./config/rq2.lcld.yaml
python -m src.run_rq2 -c ./config/rq2.botnet.yaml
#
## RQ3
python -m src.run_rq3 -c ./config/rq3.lcld.yaml
python -m src.run_rq3 -c ./config/rq3.botnet.yaml

# RQ4
python -m src.experiments.united.04_moeva -c ./config/rq4.lcld.moeva.yaml
python -m src.experiments.united.04_moeva -c ./config/rq4.lcld.moeva_augmented.yaml

## Supplementary material 1
python -m src.run_rq1 -c ./config/sm1.1.lcld.yaml
python -m src.run_rq1 -c ./config/sm1.2.lcld.yaml
python -m src.run_rq1 -c ./config/sm1.1.botnet.yaml
python -m src.run_rq1 -c ./config/sm1.2.botnet.yaml
