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
artifacts are any files, objects, or binary outputs generated during an experiment run that you need to save, track, and associate with that specific training iteration.

While metrics (like accuracy or loss) and parameters (like learning rate) are small text-based data points stored in your relational metadata database (mlflow.db), artifacts are the heavy files and physical payloads that represent the actual deliverables of your code.

Files generated during your experiment:
- Model weights and checkpoints.
- Training graphs.
- Confusion matrices.
- Evaluation reports.
- Configuration files.
- Source Code Snapshots: A copy of the exact Python scripts or git commit hash state used to execute the run, ensuring complete reproducibility.

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



##### infer_signature():
describes what data your model expects as input and what it returns as output.

In your script:
signature = infer_signature(X_train, model.predict(X_train))
MLflow examines both arguments and records:
- Input: rows containing four floating-point values—the four Iris flower measurements.
- Output: an integer class prediction for each row (0, 1, or 2).
Passing signature=signature when logging the model saves this description in its MLmodel file.

This helps in two ways:
1. Understanding the model: someone loading it can inspect the expected input structure.
2. Validating predictions: MLflow’s pyfunc interface and serving endpoint use the signature to validate inputs, such as detecting the wrong number of features.
It doesn’t improve accuracy or train the model. It records the model’s input/output contract.

Because your inputs are NumPy arrays, the signature captures their shape and data type. Using a pandas DataFrame would also capture feature names.


### Useful things to use
- What to record	
- Why it helps	
- Your Iris example

- More evaluation metrics	
- Accuracy can hide class-specific mistakes	
- Macro F1, precision, recall

- Training and validation scores	
- Helps identify overfitting	
- High training accuracy but lower validation accuracy

- Evaluation artifacts	
- Makes errors easier to inspect	
- Confusion matrix, classification report

- Data information	
- Identifies which data produced the result	
- Dataset source/version, class counts, split settings

- Code version	
- Connects the run to its implementation	
- Git commit hash

- Environment	
- Helps reproduce or reload the model	
- Python and library versions

- Model interface	
- Describes valid prediction inputs	
- Signature and input example

- Tags and descriptions	
- Explains the purpose of the run	
- model_type=random_forest, “Testing shallow trees”

- Runtime information	
- Helps compare computational cost	
- Training duration, prediction latency



##### ikv


We want to answer:
For this prediction task, which training approach should I choose, and what evidence supports that choice?

MLflow stores the history of how you answered that question. It doesn’t decide how you should organize your work, train your models, or choose the winner. You define the experiment structure; MLflow records it.

An experiment is the collection of work around a question.

For your project, the question is: “How should I classify Iris flowers?”

So iris-classification is a sensible experiment name. Inside that experiment, you can try Random Forest, Logistic Regression, SVM, different hyperparameters, and different preprocessing approaches.

You generally wouldn’t create a separate experiment for every algorithm, because you want to compare those algorithms while solving the same task.

An experiment is also not a single execution of your script. It is the container that remains across many executions.

A run is one recorded attempt or execution.

Suppose you train Logistic Regression with a particular configuration and evaluate it. You want a record containing:
- Which algorithm and configuration you used.
- Which data and code you used.
- Its evaluation scores.
- The fitted model and any reports.
That record is a run.

Then you change its regularization setting and train again. That is another run. You try Random Forest. Another run.

The algorithm name doesn’t define the run’s identity. A run represents a particular attempt with particular settings and results. Repeating the same configuration tomorrow still creates another run.

For a simple learning project, this is enough:
Experiment: iris-classification
│
├── Run: Logistic Regression, configuration A
├── Run: Logistic Regression, configuration B
├── Run: Random Forest, configuration A
├── Run: Random Forest, configuration B
└── Run: SVM, configuration A
You do not need parent and child runs to compare algorithms. Ordinary runs in the same experiment are sufficient.

Parent and child runs become useful when one execution coordinates several attempts.

Suppose you run one comparison script that tries three algorithms, with two configurations for each. You may want to keep those six attempts together because they belong to the same comparison: same data split, same evaluation procedure, same code execution.

Then the structure can be:
Experiment: iris-classification
│
├── Parent run: algorithm-comparison-001
│   ├── Child: Logistic Regression, configuration A
│   ├── Child: Logistic Regression, configuration B
│   ├── Child: Random Forest, configuration A
│   ├── Child: Random Forest, configuration B
│   ├── Child: SVM, configuration A
│   └── Child: SVM, configuration B
│
└── Parent run: algorithm-comparison-002
    ├── Child: Logistic Regression, configuration C
    ├── Child: Random Forest, configuration C
    └── Child: SVM, configuration C
Here, the parent means “this comparison session.” Each child means “one candidate tested during that session.”

A parent isn’t inherently a model. It can hold the search settings, dataset information, overall duration, winning candidate ID, and summary results.

A child isn’t a special kind of training algorithm. It is an ordinary run linked to another run. The hierarchy gives your records context.

In your current file, the parent coordinates a search, and each child represents one Random Forest configuration. Trying different algorithms extends the meaning of a candidate from “these Random Forest settings” to “this algorithm, preprocessing, and settings.”

That last part matters. You are usually comparing complete training approaches:
Candidate	What you actually train
Logistic Regression	Scaling followed by Logistic Regression
SVM	Scaling followed by SVM
Random Forest	Random Forest, usually without scaling


Different algorithms can need different preprocessing. Fair comparison means giving each approach appropriate preprocessing while keeping the evaluation conditions consistent.

The hierarchy organizes your work. The evaluation procedure makes the comparison meaningful.

For your Iris project, the procedure should be:
1. Reserve the test set once. Keep the same 30 test flowers aside.
2. Compare candidates using the remaining 120 flowers. Use the same cross-validation folds for all candidates.
3. Record validation results in each candidate run. For example, mean macro F1 and its variation across folds.
4. Choose a candidate using a metric you decided beforehand.
5. Train that selected configuration on all 120 training flowers.
6. Evaluate the selected model on the reserved test set.
The test score answers, “How well does my selected approach generalize to these held-out examples?” It should not become another score you repeatedly use to choose algorithms.

Cross-validation introduces another distinction: a candidate run can contain several model fits. With five-fold cross-validation, one candidate involves five fitted models. You can store their scores together in one run. You don’t need a child run for every fold.

Finally, you decide where the saved models belong.

For a small comparison, you can save a fitted model in every candidate run if you want to reload each approach later. Or you can save candidate scores and only save the final selected model. Your current script follows the second approach and saves the winner in the parent run. That is a design choice, not an MLflow requirement.

For you, I’d keep one iris-classification experiment, use one parent per complete comparison, and use one child per algorithm/configuration candidate. That gives each level a clear meaning:
Experiment: What problem am I studying?
Parent run: Which comparison session was this?
Child run: Which specific approach did I test?
Metrics and artifacts: What happened, and what evidence did it produce?


#### so if we run same file twice, what will change? will the database get updated?







1:24 AM






Yes. Each execution adds new records to the same database, provided you use the same tracking location.

For your current script, running it twice produces:
Experiment: iris-classification
│
├── Parent run from execution 1
│   └── Four child runs
│
└── Parent run from execution 2
    └── Four child runs
That means 10 runs total: two parents and eight children.
What	Second execution
Experiment	Reuses iris-classification
Run names	Repeat
Run IDs	New unique IDs
Previous results	Remain unchanged
Parameters	Recorded again in the new runs
Scores	Usually match with unchanged code, data, and environment because the seeds are fixed
Durations and timestamps	Change
Reports, plots, saved model	Saved again as new artifacts/model records


The database stores the new run metadata, parameters, metrics, and artifact references. The actual model and plot files are stored separately in the artifact storage.

MLflow keeps a history; it doesn’t overwrite the previous run just because the filename or run name is the same.


# Bikie
The collector already knows what to download and how to preserve it. Now we need somewhere durable to store the data and something that runs the collector automatically.

The proposed flow is:
EventBridge Scheduler — every five minutes
            ↓
Lambda — runs our collector
            ↓
Downloads the three public feeds
            ↓
Temporarily saves the collection in /tmp
            ↓
Uploads the files to a private S3 bucket
            ↓
Uploads manifest.json LAST
Each component has one job:
Component	Why we need it
S3	Keeps collections after the collector finishes
Lambda	Runs the collector without a continuously running server
EventBridge Scheduler	Starts Lambda every five minutes
IAM role	Gives Lambda permission to write to our bucket without putting AWS keys in the code
CloudWatch logs	Shows whether collection succeeded or failed


Why /tmp?

Lambda provides temporary working storage. Our existing collector can build a collection there, then we upload it to S3. Temporary files aren’t our permanent archive—S3 is.

S3 would contain:
Private bucket/
└── raw/
    └── collected_date=YYYY-MM-DD/
        ├── collection-A/
        │   ├── discovery.json.gz
        │   ├── station_status.json.gz
        │   ├── station_information.json.gz
        │   └── manifest.json
        └── collection-B/
            └── ...
Why upload the manifest last?

Uploading several S3 files is not one atomic operation. An upload could fail halfway through.

We’ll establish this rule:
A collection is ready for processing only when its manifest exists.

Lambda uploads the three data files first, then the manifest. If an earlier upload fails, it must not publish the manifest. Later stages can ignore incomplete collections.

We’ll implement this in two small stages:
1. Manual cloud collection: create the bucket and Lambda, invoke it once, and verify the saved files and checksums.
2. Automatic collection: enable the five-minute schedule, check repeated executions, and add failure alerts.
We’ll also prepare a way to disable the schedule and remove the resources, keeping your credit budget in view.

This milestone is complete when: your laptop can be off, new completed collections appear in S3, and we can distinguish a successful collection from a failed one.

After that, we’ll validate the accumulated station data and build versioned training datasets. We won’t start model training yet.

I’ll wait for your instruction before coding or creating AWS resources.


#### fmkrs 1
Now we’ll prepare our collector to run inside Lambda and upload its results to S3.

Currently, the collector saves files on your laptop. Lambda needs two additions:
1. An entry point called a handler: AWS calls this function when Lambda starts.
2. S3 upload logic: sends the completed collection to your bucket.
The flow will be:
You click Test in Lambda
        ↓
AWS calls our handler
        ↓
Handler runs the existing collector
        ↓
Collection is saved temporarily in /tmp
        ↓
Upload the three compressed files to S3
        ↓
Upload manifest.json LAST
        ↓
Return the saved S3 location
We’ll create a deployment ZIP containing the collector, handler, and required dependencies. You’ll upload that ZIP to Lambda.

First we’ll test one manual collection. Once the four files appear correctly in S3, we’ll add the five-minute schedule.


#### fmkrs 2

Let’s start with how your code starts running.

On your laptop, you run:
uv run python -m mlops_llmops.citibike.collector
Python opens collector.py, reaches this part, and calls main():
if __name__ == "__main__":
    main()
On AWS, you won’t type that terminal command. Instead, you upload the code and tell AWS:
“When I click Test, call this particular Python function.”

That starting Python function is called the handler. It’s an ordinary function—the name describes its job.

It will coordinate these steps:
AWS calls our starting function
            ↓
It calls the existing collector
            ↓
Collector downloads and saves the files temporarily
            ↓
Starting function uploads those files to S3
            ↓
It returns a message saying where they were saved
Why add this function? Your existing collector handles downloading and saving locally. The new function connects that work to AWS and adds the S3 upload step.

We’ll reuse the collector. We don’t need to rewrite its downloading logic.

Also, these two terms mean different things:
- Lambda function: the AWS resource you created named citibike-collector. It holds your uploaded code, settings, and permissions.
- Handler: the Python function inside that uploaded code that AWS calls first.
For now, the key idea is: we need to tell AWS where execution starts, just as your laptop currently starts through main().

#### fmkrs 3
“Temp” means temporary storage: a working area we use while a task runs, then clean up.

We use it for two different reasons.

In collector.py: prepare the collection before publishing it
Create temporary directory
        ↓
Write all three data files and the manifest
        ↓
Rename it to the final collection directory
The rename happens only after everything is written successfully. If writing fails, the temporary directory is cleaned up, so we don’t leave a half-written collection looking complete.

On success, renaming preserves the files in their final location.

In lambda_handler.py: hold files until they reach S3

Lambda provides writable temporary storage at /tmp.
Create our working directory under /tmp
        ↓
Collector builds a complete collection inside it
        ↓
Upload the collection to S3
        ↓
Remove our working directory
These fit together:
/tmp/citibike-<random>/             ← handler's working directory
└── collected_date=.../
    └── .collect-<random>/         ← collector prepares files here
        ↓ renamed when complete
    └── <collection-id>/          ← uploaded to S3, then cleaned up
The thought process: prepare files safely, preserve the completed result in its permanent destination, and remove working files afterward.

TemporaryDirectory handles cleanup when its with block exits normally or through an exception. A forced shutdown can interrupt cleanup; Lambda’s /tmp is therefore never our permanent archive.