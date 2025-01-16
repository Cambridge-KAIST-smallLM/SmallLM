![Auto Assign](https://github.com/Cambridge-KAIST-smallLM/demo-repository/actions/workflows/auto-assign.yml/badge.svg)

![Proof HTML](https://github.com/Cambridge-KAIST-smallLM/demo-repository/actions/workflows/proof-html.yml/badge.svg)

Install 

```bash
pip install -r requirements.txt
```

5 Experiments needed to run forward

(config_dict file in recipes/config directory, bash file in run directory, yaml file in recipes/sample directory)

1.  (config_tied_162m.json, train.sh, 14m.yaml)
2.  (config_untied_162m.json, train.sh, 14m.yaml)
3.  (config_tied_123m.json, train.sh, 14m.yaml)
4.  (config_tied_123m.json, zloss.sh, 14m.yaml)

When you run each experiments, make sure that you change the name of output directory in 14m.yaml or 14m_proposed.yaml file. 

save_steps for 1000 steps (in 14m.yaml or 14m_proposed.yaml)

Also, you should change config file name to config.json when running experiment. ex) config_tied_135m.json -> config.json

After fininsh running experiments, please run token_embedding_statistics.py. When running algorithm only check line 63, output dir address

each device can take 64 batch size so please use accum: 4 so that 64 * 4 * 8 = 2048


   
