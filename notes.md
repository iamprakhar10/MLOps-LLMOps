# ML Flow
The hidden assumption: 
"I will remember how this model was created"

###### What's an ML Experiment
code
data
parameteres
randomness
environment

###### What git captures(and what it doesn't)
code changes - yes
parameters - No
metrics - No
artifacts - No
environment - No
The core problem: ML systems lose decision history

###### Why this becomes dangerous in production
Model retraining
data drift
team changes
Infrastructure changes

###### The question every production team msut answer
"Why is this model in production"

###### What happens without tracking
No reproducibility
no auditability
no safe rollbacks
no trust

###### Common Early-Stage Excuses
We'll clean it up later
this is just research
Tracking slows us down

###### Wha tbreaks first at scale
Folder based Tracking
Spreadsheet comparisions
Memory-based decisions

###### This is not. atooling problem
This is a system and discipline problem


##### MLflow has major useful components

MLflow Tracking:	Records experiments, parameters, metrics and artifacts.
MLflow Projects:	Helps package and reproduce ML code and execution environments.
MLflow Models:	Provides a standard format for packaging models.
Model Registry:	Manages model versions and their lifecycle.
MLflow Tracing:	Tracks what happens inside LLM applications and AI agents.
Evaluation:	Evaluates model and LLM application quality.


#### MLflow tracking
MLflow organizes your work using three concepts:

Experiment

A collection of related experiments for a particular task.

Run 1

Learning rate = 0.0002, accuracy = 72%

Run 2

Learning rate = 0.0001, accuracy = 76%

Run 3

Learning rate = 0.00005, accuracy = 81%


One experiment contains multiple runs. Each run represents one execution of your training or evaluation code.
Every run can store four important types of information:

1. Parameters

The configuration you used:
learning_rate = 0.0001
batch_size = 8
epochs = 3
model_name = "llama-3"


2. Metrics

The numerical results of your experiment:
accuracy = 0.84
training_loss = 0.21
validation_loss = 0.28

3. Artifacts

Files generated during your experiment:
- Model weights and checkpoints.
- Training graphs.
- Confusion matrices.
- Evaluation reports.
- Configuration files.

4. Model

The actual trained model, which can be saved and managed for future use.

MLflow's Tracking API records these things and provides a UI to compare different runs


#### Let's see how it works in actual Python code
Suppose you're fine-tuning an LLM.

Without MLflow, you might write:
learning_rate = 0.0001
batch_size = 8
epochs = 3

-# Train your model
model = train_model(
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs
)
                
-# Evaluate your model
accuracy = evaluate_model(model)
print(accuracy)



The problem is that the results are not automatically organized into a searchable experiment history.

Now let's integrate MLflow.

First, install it:
pip install mlflow


Then:
import mlflow

-# Give your experiment a name
mlflow.set_experiment("LLM-Fine-Tuning")

-# Start a new experiment run
with mlflow.start_run():
    learning_rate = 0.0001    
    batch_size = 8    
    epochs = 3   

    # Log hyperparameters    
    mlflow.log_param("learning_rate", learning_rate)    
    mlflow.log_param("batch_size", batch_size)    
    mlflow.log_param("epochs", epochs)    
    
    # Train your model    
    model = train_model(
                learning_rate=learning_rate,        
                batch_size=batch_size,        
                epochs=epochs    
                )    
    # Evaluate    
    accuracy = evaluate_model(model)    

    # Log evaluation results    
    mlflow.log_metric("accuracy", accuracy)    
    
    # Save training output files    
    mlflow.log_artifacts("./training_outputs")    
    
    print("Experiment completed!")



That's it!

You have integrated MLflow into your training workflow.

Now every time you execute this script:
1. MLflow creates a new run.
2. It records your hyperparameters.
3. It records your evaluation metrics.
4. It stores your output files.
5. It makes the results available through its UI.

To open the UI, run:
mlflow ui


Then visit:

http://localhost:5000

You can visually compare runs, sort by accuracy, inspect parameters, and find the best-performing experiment.

Notice something important: MLflow does not train your model. Your PyTorch, Transformers, or scikit-learn code does that.

MLflow records and organizes what happens during training.

That distinction is fundamental.


#### MLflow Model Registry - managing models in production
Let's say you've trained 100 versions of your model.

After evaluation, you select the best one.

Now what?

You need to:
- Save the model somewhere reliable.
- Track which version is best.
- Know which experiment produced it.
- Make it available to your deployment system.
- Roll back if a newer version performs poorly.
This is where the Model Registry comes in.

Think of it as GitHub-style version management, but for trained ML models.

For example:
Model Registry: Customer-Support-LLM

Version 1
Archived

Produced by Experiment 1 · Accuracy: 72%

Version 2
Previous

Produced by Experiment 3 · Accuracy: 81%

Version 3
Champion

Produced by Experiment 5 · Accuracy: 84%

Illustrative example of model version management.
An important concept here is model lineage.

Suppose someone asks:

"Where did this production model come from?"

With MLflow, you can trace it back to:
- The training experiment.
- The run that produced it.
- Its parameters.
- Its evaluation metrics.
- Its associated artifacts.
You can also use aliases such as @champion to identify the model version currently selected for production.

If Version 4 becomes better, you can move the champion alias to Version 4 without changing the model reference used by your application


#### MLflow in the modern AI/LLM world (very important)
Now we get to something particularly relevant to AI engineers.

MLflow has evolved beyond traditional ML experiment tracking.

With MLflow 3, it also supports GenAI observability, tracing, evaluation, and prompt management.

Let's understand why.

Imagine you've built a RAG application:
User Question

Retriever
Fetch relevant documents
Prompt Construction
Question + retrieved context
LLM
Generate answer
Final Answer


Now imagine the user complains:

"The answer is incorrect."

Where is the problem?
- Did the retriever fetch irrelevant documents?
- Was the prompt constructed incorrectly?
- Did the LLM hallucinate?
- Did the LLM receive the wrong context?
- Did the model take too long?
- Did it consume too many tokens?
If you only log the final answer, debugging becomes difficult.
This is where MLflow Tracing helps.
Tracing records what happens at each step of your AI application.

A trace might contain:
Component	Information captured
Retriever: 	Input query and retrieved documents
Prompt: 	Prompt sent to the LLM
LLM	:  Input, output, latency and token usage
Tool calls: 	Tool inputs, outputs and execution details
Final response:	The final result returned to the user

Now you can inspect exactly where something went wrong.

MLflow also supports evaluating LLM applications using predefined or custom scorers, including LLM-as-a-judge evaluation.

You can assess:
- Answer correctness.
- Relevance.
- Hallucinations.
- Safety.
- Response quality.
It can also evaluate production traces to identify quality problems in real user traffic. 



This makes MLflow relevant not just to traditional ML engineers, but also to people building RAG systems, LLM applications, and AI agents.

Example: Adding tracing to your Python application
MLflow makes tracing relatively straightforward:


import mlflow

mlflow.set_experiment("RAG-Application")

@mlflow.trace
def retrieve_documents(query):
    # Your retrieval logic    
    return retriever.invoke(query)
    
@mlflow.trace
def generate_answer(query, documents):
    # Your LLM generation logic    
    return llm.invoke(query, context=documents)
    
@mlflow.trace
def rag_pipeline(query):
    documents = retrieve_documents(query)    
    answer = generate_answer(query, documents)    
    return answer
    
response = rag_pipeline("What is attention?")



MLflow can capture the execution of these functions and their nested operations as traces.

You can inspect those traces in its UI.

This is particularly useful when your application has multiple components and you need to debug failures


#### 1. The Backend REST API (The Data Receiver)
When your Python script or Jupyter notebook runs mlflow.log_metric("accuracy", 0.95) or mlflow.log_artifact(), it isn't writing directly to a static file on disk (unless you are running MLflow in purely local file-store mode without a server).

Instead, your client code is acting as an HTTP client, making REST API calls (POST /api/2.0/mlflow/runs/log-parameter, etc.) over HTTP to that server running on port 5000. The server receives those requests and writes them to a backend database (like SQLite or PostgreSQL) and an artifact store (like a local folder or an S3 bucket).


#### The Frontend UI (The Visual Dashboard)
The web interface you see in your browser (built with a React frontend) is served by that exact same server process.

When you open [http://127.0.0.1:5000](http://127.0.0.1:5000) in Google Chrome, the frontend app loads in your browser.

As you click around to compare runs, view loss curves, or check model versions, your browser's frontend JavaScript makes asynchronous API requests back to those same REST endpoints to fetch and render the experiment data.


#### So every experiment is getting stored in mlflow.db?

Yes, exactly—by default!

When you run an MLflow tracking server locally without specifying a custom database configuration, it automatically defaults to storing all experiment metadata in a local SQLite database named mlflow.db (created right in your current working directory).

However, it is important to distinguish what goes into mlflow.db versus what goes elsewhere:

1. What goes into mlflow.db (Metadata Store)
- The SQLite file handles metadata only:
- Experiment names and IDs
- Run IDs, start/end timestamps, statuses, and tags
- Hyperparameters and configuration settings (mlflow.log_param())
- Evaluation metrics (mlflow.log_metric())
- Model Registry tracking data (versions, stages)

2. What does NOT go into mlflow.db (Artifact Store)

Heavy files—such as trained model weights (like serialized .pkl or .safetensors files), plots, dataset snapshots, or large text logs—are too bulky to store efficiently inside a relational database row.

Instead, those heavy files are saved separately in an Artifact Store (which defaults to a local folder like ./mlartifacts or ./mlruns on your machine, or an S3/GCS bucket if you are configured for production).

When you log a model artifact, two things happen:

The physical file is saved to the artifact storage directory.

A pointer record (the file path or URI pointing to that artifact) is written to mlflow.db.

Scaling Beyond mlflow.db
While SQLite (mlflow.db) is fantastic for local development, solo projects, and testing, enterprise environments switch the backend store URI (--backend-store-uri) to a production-grade relational database like PostgreSQL or MySQL to handle high concurrent writes from multiple team members or CI/CD pipelines.