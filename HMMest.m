function [estStates, hmmF] = HMMest(choiceSeq)

% Hongli Wang, 2/26/2019
% inferring the explore/exploit state for mouse behavior task using hidden
% Markov model
% input:        choiceSeq: a sequence of choices (1=left, 2=right)
% output:       estStates: possible states sequence estimated using MLE
%               hmmF: a struct contains parameters in the model

%construct the model first
nstates = 3;
hmm.nObsStates = 2;
emission0 = [1/2, 1/2; 1, 0; 0, 1];
hmm.emission = tabularCpdCreate(emission0);
hmm.nstates = nstates;
hmm.pi = [1, 0, 0];
trans0 = [1/3,1/3,1/3; 1/3, 2/3, 0; 1/3, 0, 2/3];
hmm.A = trans0;
hmm.type = 'discrete';
fprintf('HMM\n');

% fit the model
hmmF = hmmFit(choiceSeq, nstates, 'discrete', 'verbose', true, ...
    'pi0', hmm.pi, 'trans0', hmm.A, 'emission0', hmm.emission);

%infer hidden state
estStates = hmmviterbi(choiceSeq, hmmF.A, hmmF.emission.T);
