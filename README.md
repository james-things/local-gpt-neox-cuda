# local-gpt-neox-cuda

 An simple example python script demonstrating a means of running GPT-NeoX-2.7B inference locally on a single GPU.
 
 1/30/23: Added fp16 loading version, which allows running text-generation without hitting memory limits on an 11GB GPU. Tested up to 1000 characters on a 1080Ti. This version of the script accepts two parameters; see the following example:

 `python gpt-neox-fp16.py -p "A good example of a prompt could be" -l 1000` - will generate 1000 characters beginning with "A good example of a prompt could be"

 Tested under Windows using Anaconda and transformers 4.24.0. Requires an installation of torch, with the appropriate cuda version as per the system's installed cuda drivers.
