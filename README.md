![Auto Assign](https://github.com/Cambridge-KAIST-smallLM/demo-repository/actions/workflows/auto-assign.yml/badge.svg)

![Proof HTML](https://github.com/Cambridge-KAIST-smallLM/demo-repository/actions/workflows/proof-html.yml/badge.svg)

Install 

```bash
pip install -r requirements.txt
```

5 Experiments needed to run forward

(config_dict file in recipes/config directory, bash file in run directory, yaml file in recipes/sample directory

1. (config_tied_135m.json, train.sh, 14m.yaml) 2. (config_untied_135m.json, train.sh, 14m.yaml) 3. (config_tied_110m.json, train.sh, 14m.yaml) 4. (config_tied_110m.json, proposed,sh, 14m_proposed.sh) 5. (config_tied_110m.json, zloss.sh, 14m.yaml)

   
