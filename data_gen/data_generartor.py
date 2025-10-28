from perturbation_generator import *
import os
import yaml
import json
from datetime import datetime

if __name__ == "__main__":
    # Please configure some parameters in the config file, especially the data source path, data file format, etc. 
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"📄 loaded config file: {config_path}")

    perturbation_config = config.get('perturbation', {})
    perturbation_method = perturbation_config.get('method', 'random')
    perturb_generator_list = []
    if perturbation_method == 'random':
        perturb_generator_list.append(RandomPerturbationGenerator(config))
    elif perturbation_method == 'semantic':
        perturb_generator_list.append(SemanticPerturbationGenerator(config))
    elif perturbation_method == 'topology':
        perturb_generator_list.append(TopologyPerturbationGenerator(config))
    elif perturbation_method == 'mixture0_1':
        perturb_generator_list.append(RandomPerturbationGenerator(config))
        perturb_generator_list.append(SemanticPerturbationGenerator(config))
    elif perturbation_method == 'mixture0_2':
        perturb_generator_list.append(RandomPerturbationGenerator(config))
        perturb_generator_list.append(TopologyPerturbationGenerator(config))
    elif perturbation_method == 'mixture1_2':
        perturb_generator_list.append(SemanticPerturbationGenerator(config))
        perturb_generator_list.append(TopologyPerturbationGenerator(config))
    elif perturbation_method == 'mixture1_2_3':
        perturb_generator_list.append(RandomPerturbationGenerator(config))
        perturb_generator_list.append(SemanticPerturbationGenerator(config))
        perturb_generator_list.append(TopologyPerturbationGenerator(config))
    else:
        raise ValueError(f"Unsupported: {perturbation_method}")

    record_dir = os.path.join(os.path.dirname(__file__), 'perturb_record')
    os.makedirs(record_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for idx, perturb_generator in enumerate(perturb_generator_list):
        perturbation_info = perturb_generator.apply_perturbation()
        
        generator_name = perturb_generator.__class__.__name__
        record_filename = f"{perturbation_method}_{generator_name}_{timestamp}.json"
        record_path = os.path.join(record_dir, record_filename)
        
        # Add log info to perturbation_info
        log_info = {
            "generator_index": f"{idx + 1}/{len(perturb_generator_list)}",
            "record_path": record_path,
            "method": perturbation_info.get('method', 'N/A'),
            "operation_count": len(perturbation_info.get('operations', [])),
            "perturbed_data_path": perturbation_info.get('perturbed_data_path', 'N/A')
        }
        perturbation_info["log_info"] = log_info
        
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(perturbation_info, f, ensure_ascii=False, indent=2)
        
    # Add summary info to a separate file
    summary_filename = f"perturbation_summary_{timestamp}.json"
    summary_path = os.path.join(record_dir, summary_filename)
    summary_info = {
        "total_generators": len(perturb_generator_list),
        "timestamp": timestamp,
        "perturbation_method": perturbation_method
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_info, f, ensure_ascii=False, indent=2)
    