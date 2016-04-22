addpath(genpath('.'));
% Import chatlog (#acceuil)
fID = fopen('chatlog.txt', 'r', 'n', 'UTF-8');
tline = fgetl(fID);
% Data structures
pseudos = {}; hour = {}; messages = {};
% Basic parsing
while ischar(tline)
    checkHead = regexp(tline, '[^ ] [0-2][0-9]:[0-5][0-9]');
    if (~isempty(checkHead))
        pseudos = [pseudos, tline(1:checkHead(1))];
        hour = [hour, tline(checkHead(1)+1:end)];
    else
        messages = [messages, tline];
    end
    tline = fgetl(fID);
end
fclose(fID);
% Now generate the message-only file
fID = fopen('messages.txt', 'w');
for m = 1:length(messages)
    if ~isempty(messages{m}) && ~(strcmp('A rejoint le canal.', messages{m}) || strcmp('A quitté le canal.', messages{m}))
        fprintf(fID, '%s\n', messages{m});
    end
end
% Integrate automatic machine translation here
% So that we can thrive on English-based models (way more accurate)
% => Generates "traducted.txt"
%% Re-arrange messages to fit paraphrase checking (+ sentence extraction)
fID = fopen('traducted.txt', 'r', 'n', 'UTF-8');
fIDExt = fopen('traducted_extended.txt', 'w');
tline = fgetl(fID);
% Data structures
messages = {};
% Retrieve all messages
while ischar(tline)
    messages = [messages, tline];
    messageExt = strsplit(tline, '.');
    for m = 1:length(messageExt)
        if ~isempty(messageExt{m}) && length(messageExt{m}) > 32
            fprintf(fIDExt, '%s\n', messageExt{m});
        end
    end
    tline = fgetl(fID);
end
fclose(fID);
fclose(fIDExt);
%% Final processed input
fID = fopen('input.txt', 'w', 'n', 'UTF-8');
fIDs = fopen('sentences.txt', 'w', 'n', 'UTF-8');
fIDl = fopen('labels.txt', 'w', 'n', 'UTF-8');
% Create binomial combinatory distribution of sentences
for m1 = 1:(length(messages)-1)
    fprintf('Processing message %d\n', m1);
    if (length(messages{m1}) < 64)
        continue;
    end
    for m2 = (m1+1):length(messages)
        if (length(messages{m2}) < 64)
            continue;
        end
        fprintf(fID, '1\n%s\n%s\n', messages{m1}, messages{m2});
        fprintf(fIDs, '%s\n%s\n', messages{m1}, messages{m2});
        fprintf(fIDl, '1\n');
    end
end
fclose(fID);
%% Test automatic paraphrase (Ng's Autoencoders)
%system('sed "1~3d" input.txt > sentences.txt');
%system('sed -n "p;N;N" input.txt > labels.txt');
%system('paraphrases/stanford-parser-full-2015-12-09/lexparser.sh sentences.txt > parsed.txt');
inputFile = 'parsed.txt';
convertStanfordParserTrees
simMat
%% -------------------
% Perform a context-based word encoding
% 100232 words in vocab
% 6162 of those have 10 prototypes each
addpath(genpath('.'));
load('wordreps.mat','We');   % load word representations
load('vocab.mat','vocab','tfidf','numEmbeddings');
dsz = length(vocab);
% load word reps used for clustering
load('wordreps_orig.mat','oWe');
% load cluster centers
load('centers.mat','centers','orig2cent');
sPad = find(strcmp('<s>',vocab));
ePad = find(strcmp('</s>',vocab));
addpath(genpath('.'));
messagesIDs = cell(length(messages), 1);
messagesPROs = cell(length(messages), 1);
% Create linguistic statistics
basicStats = containers.Map({'Poyoyoyo'}, {0});
semanticStats = zeros(5000, 1);
semanticEquiv = cell(5000, 1);
semanticStatsPRO = zeros(5000, 1);
semanticEquivPRO = cell(5000, 1);
for m = 1:length(messages)
    fprintf('Parsing message %d\n', m);
    % input: a tokenized sentence, one word per cell
    inputMsg = strsplit(messages{m}, ' ');
    % output: corresponding vocab id and prototype number for each word
    ids = ones(1,length(inputMsg));
    for i = 1:length(inputMsg)
        if basicStats.isKey(inputMsg{i})
            basicStats(inputMsg{i}) = basicStats(inputMsg{i}) + 1;
        else
            basicStats(inputMsg{i}) = 1;
        end
        id = find(strcmp(lower(inputMsg{i}),vocab));
        if isempty(id) %&& regexp(inputMsg{i},'^[-+]?[0-9]*\.?[0-9]+')
            id = find(strcmp(regexprep(inputMsg{i},'[0-9]','DG'),vocab));
            if isempty(id)
                id = find(strcmp('NNNUMMM',vocab));
            end
        end
        if isempty(id) || id > 5000  % only use the top 5000 most frequent words
            id = 1;
        end
        ids(i) = id;
    end
    ids = [repmat(sPad,1,5) ids repmat(ePad,1,5)]; % pad
    pros = ones(1,length(ids)); % prototype numbers
    for i = 5+1:length(ids)-5
        if orig2cent(ids(i)) == 0
            pros(i) = 1;
        else
            c = squeeze(centers(:,orig2cent(ids(i)),:));    
            % compute the context representation, which is a tf-idf weighted average
            % of the representations of the context words within a 10-word window
            contexts = ids([i-5:i-1 i+1:i+5])';
            unigrams = contexts;
            tf = sparse(unigrams(:),ones(size(unigrams(:))),tfidf(unigrams(:)),dsz,size(unigrams,2));
            tf = bsxfun(@rdivide,tf,sum(tf));
            contexts = reshape(oWe,50,[]) * tf;
            % find cluster by choosing the cluster center rep. that's closest to the 
            % context rep. we use cosine distance here.
            dist = slmetric_pw(contexts,c,'corrdist');
            [~,cluster] = min(dist,[],2);
            pros(i) = cluster;
        end
        semanticStats(ids(i)) = semanticStats(ids(i)) + 1;
        semanticEquiv{ids(i)} = [semanticEquiv{ids(i)}, inputMsg{i-5}];
        semanticStatsPRO(ids(i)) = semanticStatsPRO(ids(i)) + 1;
        semanticEquivPRO{ids(i)} = [semanticEquivPRO{ids(i)}, inputMsg{i-5}];
    end
    messagesIDs{m} = ids(6:end-5);
    messagesPROs{m} = pros(6:end-5);
end
%%
basicValues = cell2mat(basicStats.values);
basicKeys = basicStats.keys;
[bVal, idS] = sort(basicValues, 'descend');
for i = 1:100
    fprintf('%d\t%s\n', basicValues(idS(i)), basicKeys{idS(i)});
end
[sStats, idS] = sort(semanticStats, 'descend');
for i = 1:10
    disp(
    fprintf('%d\t%s\n', semanticStats(idS(i)), strjoin(unique(semanticEquiv{ids(i)})));
end
[sStats, idS] = sort(semanticStatsPRO, 'descend');
for i = 1:10
    fprintf('%d\t%s\n', semanticStatsPRO(idS(i)), strjoin(unique(semanticEquivPRO{ids(i)})));
end
%% Now for the real deal ... We have to cross-correlate messages
distMatrix = zeros(length(messages), length(messages));
for m1 = 1:(length(messages)-1)
    fprintf('Seeding message %d\n', m1);
    for m2 = (m1+1):length(messages)
        coOccur = 0;
        for i = 1:length(messagesIDs{m1})
            for j = 1:length(messagesIDs{m2})
                if (messagesIDs{m1}(i) == messagesIDs{m2}(j) && messagesIDs{m1}(i) ~= 1 && messagesIDs{m2}(j) ~= 1 && messagesIDs{m1}(i) ~= 770 && messagesIDs{m2}(j) ~= 770), coOccur = coOccur + 1; end
            end
        end
        coOccur = coOccur / (length(messagesIDs{m1}) + length(messagesIDs{m2}));
        %disp(coOccur);
        if (coOccur > 0.45)
            if ((length(messagesIDs{m1}) == length(messagesIDs{m2})) && ((messagesIDs{m1} - messagesIDs{m2}) == 0))
                continue;
            end
            disp(coOccur);
            disp(m1);
            disp(m2);
            disp(messages{m1});
            disp(messages{m2});
            %disp(messagesIDs{m1});
            %disp(messagesIDs{m2});
        end
    end
end