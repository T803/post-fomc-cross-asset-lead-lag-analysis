Author: Tania E. Kuoh
Date: August 2026


This repository contains all code files that were used to carry out my investigation of post FOMC cross-asset lead-lag relationships. The focus was on equities (specifically the S&P 500) and FX (specifically USDJPY) but the different files/functions should be flexible enough to accommodate other types of analyses.

## Getting Started

- Clone the repository locally using git clone https://github.com/T803/post-fomc-cross-asset-lead-lag-analysis.git

- Create your virtual environment from the project's root using pipenv install in the terminal

- Sync up the dependencies using pipenv sync

Note: If you haven't installed pipenv, you need to run the following in the terminal: python3 -m pip install pipenv

## Adding/Removing Dependencies

You can add/remove dependencies directly in the Pipfile. Once that is done, run pipenv lock to generate a new Pipfile.lock, you may verify that the new Pipfile.lock is up to date by running pipenv verify. You can then push your updated Pipfile and Pipfile.lock to the remote branch.

Note: Users will need to pull your changes and run pipenv sync for it to carry over in their virtual environments.
