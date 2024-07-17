
import os
import sys
import prov
import prov.model as prov
from datetime import datetime
import getpass
import subprocess
import warnings

from ..constants import PROV4ML_DATA
from ..datamodel.attribute_type import Prov4MLAttribute
from ..datamodel.artifact_data import artifact_is_pytorch_model
from ..provenance.context import Context
from ..utils.funcs import get_global_rank

def calculate_energy_consumption(doc, ctx, epochs, timestamps, values):
    energy = 0
    for i in range(1, len(epochs)):
        energy += (timestamps[i] - timestamps[i-1]) * values[i]

    if not doc.get_record(f'energy_consumption_{ctx}'):
        metric_entity = doc.entity(f'energy_consumption_{ctx}',{
            'prov-ml:type':Prov4MLAttribute.get_attr('Metric'),
            'prov-ml:name':Prov4MLAttribute.get_attr("energy_consumption"),
            'prov-ml:context':Prov4MLAttribute.get_attr(ctx),
            'prov-ml:source':Prov4MLAttribute.get_source_from_kind("gpu_power_usage"),
        })
    else:
        metric_entity = doc.get_record(f'energy_consumption_{ctx}')[0]

    for epoch in epochs: 
        doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',identifier=f'energy_consumption_train_{epoch}_gen')
    
    metric_entity.add_attributes({
        'prov-ml:metric_epoch_list': Prov4MLAttribute.get_attr(epochs), 
        'prov-ml:metric_value_list': Prov4MLAttribute.get_attr(energy),
        'prov-ml:metric_timestamp_list': Prov4MLAttribute.get_attr(timestamps),
        'prov-ml:context': Prov4MLAttribute.get_attr(ctx),
    })
    

def save_metric_from_file(metric_file, name : str, ctx:Context, doc:prov.ProvDocument, run_activity):
    with open(os.path.join(PROV4ML_DATA.TMP_DIR, metric_file), 'r') as f:
            lines = f.readlines()
            source = lines[0].split(',')[2]

            if not doc.get_record(f'{name}_{ctx}'):
                metric_entity = doc.entity(f'{name}_{ctx}',{
                    'prov-ml:type':Prov4MLAttribute.get_attr('Metric'),
                    'prov-ml:name':Prov4MLAttribute.get_attr(name),
                    'prov-ml:context':Prov4MLAttribute.get_attr(ctx),
                    'prov-ml:source':Prov4MLAttribute.get_source_from_kind(source),
                })
            else:
                metric_entity = doc.get_record(f'{name}_{ctx}')[0]

            metric_epoch_data = {}
            for line in lines[1:]:
                epoch, value, timestamp = line.split(',')
                epoch = int(epoch)
                value = float(value)
                timestamp = int(timestamp)
                if int(epoch) not in metric_epoch_data:
                    metric_epoch_data[epoch] = []
                metric_epoch_data[epoch].append((value, timestamp))

            for epoch in metric_epoch_data.keys():
                if ctx == Context.TRAINING: 
                    if not doc.get_record(f'epoch_{epoch}'):
                        train_activity=doc.activity(f'epoch_{epoch}',other_attributes={
                            "prov-ml:type": Prov4MLAttribute.get_attr("TrainingExecution")
                        })
                        doc.wasStartedBy(train_activity,run_activity)

                    doc.wasGeneratedBy(metric_entity,f'epoch_{epoch}',identifier=f'{name}_train_{epoch}_gen')
                    
                elif ctx == Context.VALIDATION:
                    val_name = f'val_epoch_{epoch}'
                    if not doc.get_record(val_name):
                        train_activity=doc.activity(val_name,other_attributes={
                            "prov-ml:type": Prov4MLAttribute.get_attr("ValidationExecution"),
                        })
                        doc.wasStartedBy(train_activity,run_activity)

                    doc.wasGeneratedBy(metric_entity,val_name,identifier=f'{name}_val_{epoch}_gen')
                    
                elif ctx == Context.EVALUATION:
                    if not doc.get_record('test'):
                        eval_activity=doc.activity('test',other_attributes={
                        "prov-ml:type": Prov4MLAttribute.get_attr("TestingExecution")})
                        doc.wasStartedBy(eval_activity,run_activity)

                    doc.wasGeneratedBy(metric_entity,'test',identifier=f'test_gen')
                
            epochs = []
            values = []
            timestamps = []
            for epoch, item_ls in metric_epoch_data.items():
                for (val, time) in item_ls:
                    epochs.append(epoch)
                    values.append(val)
                    timestamps.append(time)

            metric_entity.add_attributes({
                'prov-ml:metric_epoch_list': Prov4MLAttribute.get_attr(epochs), 
                'prov-ml:metric_value_list': Prov4MLAttribute.get_attr(values),
                'prov-ml:metric_timestamp_list': Prov4MLAttribute.get_attr(timestamps),
                'prov-ml:context': Prov4MLAttribute.get_attr(ctx),
            })

def create_prov_document() -> prov.ProvDocument:
    """
    Generates the first level of provenance for a given run.

    Args:
        run (Run): The run object.
        doc (prov.ProvDocument): The provenance document.

    Returns:
        prov.ProvDocument: The provenance document.
    """
    doc = prov.ProvDocument()

    #set namespaces
    doc.set_default_namespace(PROV4ML_DATA.USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('prov-ml', 'prov-ml')

    run_entity = doc.entity(f'{PROV4ML_DATA.EXPERIMENT_NAME}',other_attributes={
        "prov-ml:provenance_path":Prov4MLAttribute.get_attr(PROV4ML_DATA.PROV_SAVE_PATH),
        "prov-ml:artifact_uri":Prov4MLAttribute.get_attr(PROV4ML_DATA.ARTIFACTS_DIR),
        "prov-ml:run_id":Prov4MLAttribute.get_attr(PROV4ML_DATA.RUN_ID),
        "prov-ml:type": Prov4MLAttribute.get_attr("LearningStage"),
        "prov-ml:user_id": Prov4MLAttribute.get_attr(getpass.getuser()),
    })

    # add python version to run entity
    run_entity.add_attributes({"prov-ml:python_version":Prov4MLAttribute.get_attr(sys.version)})

    # check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        env_reqs = open("requirements.txt", "r").read()        
        run_entity.add_attributes({"prov-ml:requirements":Prov4MLAttribute.get_attr(env_reqs)})

        
    global_rank = get_global_rank()
    if global_rank is not None and os.getenv("SLURM_PROCID") is not None:
        node_rank = os.getenv("SLURM_NODEID", None)
        local_rank = os.getenv("SLURM_LOCALID", None) 
        run_entity.add_attributes({
            "prov-ml:global_rank":Prov4MLAttribute.get_attr(global_rank),
            "prov-ml:local_rank":Prov4MLAttribute.get_attr(local_rank),
            "prov-ml:node_rank":Prov4MLAttribute.get_attr(node_rank),
        })

    run_activity = doc.activity(f'{PROV4ML_DATA.EXPERIMENT_NAME}_execution', other_attributes={
        'prov-ml:type': Prov4MLAttribute.get_attr("LearningExecution"),
    })
        #experiment entity generation
    experiment = doc.entity(PROV4ML_DATA.EXPERIMENT_NAME,other_attributes={
        "prov-ml:type": Prov4MLAttribute.get_attr("Experiment"),
        "prov-ml:experiment_name": Prov4MLAttribute.get_attr(PROV4ML_DATA.EXPERIMENT_NAME),
    })

    user_ag = doc.agent(f'{getpass.getuser()}')
    doc.wasAssociatedWith(f'{PROV4ML_DATA.EXPERIMENT_NAME}_execution',user_ag)
    doc.entity('source_code',{
        "prov-ml:type": Prov4MLAttribute.get_attr("SourceCode"),
        "prov-ml:source_name": Prov4MLAttribute.get_attr(__file__.split('/')[-1]),
        "prov-ml:source_type": Prov4MLAttribute.get_attr("LOCAL") if global_rank is None else Prov4MLAttribute.get_attr("SLURM"),
    })

    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        doc.activity('commit',other_attributes={"prov-ml:source_git_commit": Prov4MLAttribute.get_attr(commit_hash)})
        doc.wasGeneratedBy('source_code','commit')
        doc.wasInformedBy(run_activity,'commit')
    except:
        warnings.warn("Git not found, skipping commit hash retrieval")
        doc.used(run_activity,'source_code')


    doc.hadMember(experiment,run_entity)
    doc.wasGeneratedBy(run_entity,run_activity)
    
    if os.path.exists(PROV4ML_DATA.TMP_DIR):
        all_metrics = os.listdir(PROV4ML_DATA.TMP_DIR) 
    else:
        all_metrics = []

    if global_rank is not None:
        all_metrics = [metric for metric in all_metrics if f"_GR{global_rank}" in metric]

    for metric_file in all_metrics:
        if global_rank is not None:
            name = "_".join(metric_file.split('_')[:-2])
            ctx = metric_file.split('_')[-2].strip()
        else: 
            name = "_".join(metric_file.split('_')[:-1])
            ctx = metric_file.split('_')[-1].replace(".txt","")
        ctx = Context.get_context_from_string(ctx)
        save_metric_from_file(metric_file, name, ctx, doc, run_activity)
                        
    for name, param in PROV4ML_DATA.parameters.items():
        if "dataset" in name: continue

        ent = doc.entity(f'{name}',{
            'prov-ml:parameter_value': Prov4MLAttribute.get_attr(param.value),
            'prov-ml:type': Prov4MLAttribute.get_attr('Parameter'),
        })
        doc.used(run_activity,ent)

    # create entity for final run statistics
    final_run_stats = doc.entity('final_run_statistics',{
        'prov-ml:type': Prov4MLAttribute.get_attr('RunStatistics'),
    })
    other_attributes = {}
    for (name, metric) in PROV4ML_DATA.cumulative_metrics.items():
        other_attributes[f'prov-ml:{name}'] = Prov4MLAttribute.get_attr(metric.current_value)
    final_run_stats.add_attributes(other_attributes)
    doc.wasGeneratedBy(final_run_stats,run_activity)

    #dataset entities generation
    ent_ds = doc.entity(f'datasets')

    for name, param in PROV4ML_DATA.parameters.items():
        if "dataset_stat" in name:
            dataset_name = name.split('_')[0] + "_dataset"
            if not doc.get_record(f'{dataset_name}'):
                ent = doc.entity(f'{dataset_name}',{'prov-ml:type': Prov4MLAttribute.get_attr('Dataset')})

                doc.used(run_activity,ent)
                doc.hadMember(ent_ds,f'{dataset_name}')

            ent = doc.get_record(f'{dataset_name}')[0]

            label = name.split('_')[-1]
            ent.add_attributes({f'prov-ml:{label}': Prov4MLAttribute.get_attr(param.value)})

    doc.wasGeneratedBy(ent_ds,run_activity)

    #model version entities generation
    model_version = PROV4ML_DATA.get_final_model()
    if model_version:
        model_entity_label = model_version.path
        modv_ent=doc.entity(model_entity_label,{
            "prov-ml:type": Prov4MLAttribute.get_attr("ModelVersion"),
            'prov-ml:creation_epoch': Prov4MLAttribute.get_attr(model_version.step),
            'prov-ml:artifact_uri': Prov4MLAttribute.get_attr(model_version.path),
            'prov-ml:creation_timestamp': Prov4MLAttribute.get_attr(datetime.fromtimestamp(model_version.creation_timestamp / 1000)),
            'prov-ml:last_modified_timestamp': Prov4MLAttribute.get_attr(datetime.fromtimestamp(model_version.last_modified_timestamp / 1000)),
        })
        doc.wasGeneratedBy(modv_ent,run_activity,identifier=f'{model_entity_label}_gen')
    
    registration_label = 'prov-ml:ModelRegistration'
    model_ser = doc.activity(registration_label)
    doc.wasInformedBy(model_ser,run_activity)
    if model_version:
        doc.wasGeneratedBy(model_entity_label,model_ser)
    else:
        model_entity_label = registration_label
    
    for artifact in PROV4ML_DATA.get_model_versions()[:-1]: 
        doc.hadMember(model_entity_label,f"{artifact.path}")    

    doc.activity("data_preparation",other_attributes={"prov-ml:type":Prov4MLAttribute.get_attr("FeatureExtractionExecution")})
    
    #artifact entities generation
    for artifact in PROV4ML_DATA.get_artifacts():
        ent=doc.entity(f'{artifact.path}',{
            'prov-ml:artifact_path': Prov4MLAttribute.get_attr(artifact.path),
        })
        #the FileInfo object stores only size and path of the artifact, specific connectors 
        # to the artifact store are needed to get other metadata
        if artifact_is_pytorch_model(artifact):
            doc.wasGeneratedBy(f"{artifact.path}", model_ser)
        else: 
            doc.wasGeneratedBy(ent,run_activity,identifier=f'{artifact.path}_gen')    

    return doc
