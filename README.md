# Test-Model-Thing (TMT)

This is a small proof-of-concept language model (not an LLM) that incorporates the following (and some smaller features as well):
* Latent-space prediction
* Internal state + recurrent trace units (RTUs)
* Byte input/output
* Continuous data streaming
* Test-time training

The model is built with MLX, so it should run fine on all Apple Silicon devices. It relies heavily on unified memory chips to achieve some things e.g. dynamic array resizing.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are very promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. However given further training and scaling up the hyperparameters this could become much more powerful. My dataset is also tiny (only a few hundred MB), so there's a lot more world knowledge that can be fed into the model.

Feel free to fork the training and benchmark code (everything is under MIT).

_I used to have a video here, but I privated it for now._

<img width="499" height="497" alt="3f7f1530-c0c7-43c4-9981-30e9023a19fb" src="https://github.com/user-attachments/assets/eb7e5a97-09b5-4a7b-9484-eb898042e9dc" />

## Training your own model

Model weights are not provided because GitHub doesn't like very large files. But, you can train your own model simply by initializing a ```venv``` and installing ```mlx```, no other libraries needed, then running ```main.py```. When you run it, you will be prompted with the mode, ```0``` being train on dataset and ```1``` being chat. You will have to configure your own dataset by modifying the code (to run dataset mode), but you should be able to run chat mode without modifying anything if you have weights already.

Once it begins training, you can safely ^C the program and it will save weights. It should also periodically save weights if I'm not mistaken. The saved weights include the internal memory so the model will remember that the next time it runs. You can launch into chat mode and the memory should carry on from whatever it was learning in training.

## How it works

This model architecture was designed in about a month by me (a solo high school dev) and some Gemini. I write READMES myself though w/o AI.

In detail, here are some of the main capabilities of the model that differ from LLMs:
* JEPA-style latent space prediction, as the decoder can be removed/disabled and the model still rolls out forward as is. The model is not trained explicitly on predicting the next byte, but rather on two separate goals (predicting the next 'thing' in latent space, and translating the current latent space vector to a byte).
* Theoretically infinite memory, as it does not have a context window and instead relies on RTUs to store internal state/memory. However it does decay old memories over time. Also I think this should be O(1) memory based on my implementation but I'm not 100% sure.
* Built-in multimodality, as the model outputs bytes (and thus should theoretically be capable of handling any binary data).
* Streaming data live, since the model only processes one byte at once at rapid pace. In fact it is completely 'blind' to everything that came before the current byte, only relying on the current processed byte and its internal memory to decide the next byte. This confirms the model is definitely learning to remember things.
* Continual training, as it keeps training on user input, training data, and its own output to improve its predictions automatically. Keep in mind the model can only output a byte (0-255) each pass anyways (in addition to updating its own state).

The two important hyperparameters are the size of the latent vector (dim) and the amount of individual state layers the latent passes through before decoding (layers). For my 4.5m test these are ```dim = 512``` and ```layers = 16```. There are some other configurations you can change but I think they are less important.

I think this probably will contribute significantly to solving continual learning and memory but I still need other people to review and verify my work! Please feel free to open GitHub issues to tell me what's wrong. If you have compute (e.g. you are a lab), feel free to fork my code and train larger models as well, with credit.

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />
