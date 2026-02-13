Download
Download via Hugging Face CLI:

# Add huggingface token (optional, avoid 429 rate limit) # export HF_TOKEN= hf download MLL-Lab/tos-data --repo-type dataset --local-dir room_data

Or use the ToS setup script which downloads automatically:

git clone --single-branch --branch release https://github.com/mll-lab-nu/Theory-of-Space.git cd Theory-of-Space source setup.sh