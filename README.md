Dear reviewers,

hi, we put our code here to offer more implementation details.  
We seperate our model into multiple modules of python files following this list:


├── aggregator.py     -> where aggregator is defined<br/>
├── config            -> configs for experiment<br/>
├── data_assign.py    -> where we define the data assignment strategy<br/>
├── data_process.py   -> process dataset, such as normalization <br/>
├── decoder.py        -> where we define different decoders<br/>
├── ellipses_noise.py -> where we generate images noises<br/>
├── encoder.py        -> where we define different encoders<br/>
├── experiment        -> where we store experiment scripts<br/>
├── __init__.py<br/>
├── logger.py         -> where we define our loggers<br/>
├── loss.py           -> where we define different types of loss functions<br/>
├── mp.py             -> where we define the Movement Primitives (ProMPs, DMPs, IDMPs)<br/>
├── ndp.py            -> where we re-implement NDP method<br/>
├── net.py            -> the commander center to train the model<br/>
├── nn_base.py        -> where we define basic neural network structures<br/>
└── util.py           -> where we define utilities functions<br/>


Our dependencies are available through pip or conda.<br/>
The PyTorch version we have tested our code is 1.10<br/>

Main dependencies:<br/>
attrdict<br/>
matplotlib <br/>
natsort <br/>
numpy <br/>
pandas<br/>
pytorch<br/>
scipy <br/>
tabulate<br/>
tensorboard <br/>
tqdm <br/>
wandb<br/>
yaml<br/>
python-mnist<br/>
