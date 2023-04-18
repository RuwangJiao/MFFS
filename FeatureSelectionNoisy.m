classdef FeatureSelectionNoisy < PROBLEM
% <multi> <real/binary> <sparse/none> 
% The feature selection problem with noise (balance accuracy calculation)
% dataNo --- 20 --- Number of dataset
% encode --- 1 --- Encoding method
% ObjectiveNo --- 2 --- Number of objectives
% NoiseLevel --- 0 --- Percentage(%) noise added 

%------------------------------- Reference --------------------------------

% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% The datasets are taken from the UCI machine learning repository in
% http://archive.ics.uci.edu/ml/index.php
%  No.    Name                              Samples Features Classes
%  1      Glass                               214      10       6
%  2      Wine                                178      13       3
%  3      Leaf                                340      14      30
%  4      Australian                          690      14       2
%  5      Zoo                                 101      17       7
%  6      Lymph                               148      18       4
%  7      Vehicle                             846      18       4
%  8      ImageSegmentation                   210      19       7
%  9      Parkinson                           195      22       2
% 10      Spect                               267      22       2
% 11      German                             1000      24       2
% 12      LedDisplay                         1000      24      10
% 13      WallRobot                          5456      24       4
% 14      WBCD                                569      30       2
% 15      GuesterPhase                       9873      32       5
% 16      Dermatology                         366      33       6
% 17      Ionosphere                          351      34       2  
% 18      Chess                              3196      36       2
% 19      Connect4                          67557      42       3
% 20      Lung                                 32      56       2
% 21      Sonar                               208      60       2
% 22      Plant                              1600      64     100
% 23      Mice                               1077      77       8
% 24      Movementlibras                      360      90      15
% 25      Hillvalley                         1212     100       2
% 26      GasSensor                         13910     128       6
% 27      Musk1                               476     166       2
% 28      USPS                               9298     256      10
% 29      Semeion                            1593     265       2
% 30      Arrhythmia                          452     278      13
% 31      LSVT                                126     310       2
% 32      lung_discrete                        73     325       7
% 33      Madelon                            2600     500       2
% 34      Isolet                             1560     617      26
% 35      Isolet5                            1559     617      25
% 36      MultipleFeatures                   2000     649      10
% 37      Gametes                            1600    1000       2
% 38      ORL                                 400    1024      40 
% 39      Yale                                165    1024      16
% 40      COIL20                             1440    1024      20
% 41      Christine                          5418    1636       2
% 42      Bioresponse                        3751    1776       2
% 43      colon                                62    2000       2
% 44      Colon                                62    2000       2
% 45      SRBCT                                83    2308       4
% 46      AR10P                               130    2400      10 
% 47      lymphoma                             96    4026       9
% 48      RELATHE                            1427    4322       2
% 49      GLIOMA                               50    4434       4
% 50      BASEHOCK                           1993    4862       2
% 51      Gisette                            7000    5000       2
% 52      Leukemia1                            72    5327       3
% 53      DLBCL                                77    5469       2
% 54      9Tumor                               60    5726       9
% 55      Brain1                               90    5920       5
% 56      Prostate-GE                         102    5966       2
% 57      CNS                                  60    7129       2
% 58      Leukemia                             72    7129       2
% 59      ALLAML                               72    7129       2 
% 60      Carcinom                            174    9182      11
% 61      nci9                                 60    9712       9
% 62      arcene                              200   10000       2
% 63      orlraws10P                          100   10304      10
% 64      Brain2                               50   10367       4
% 65      CLL-SUB-111                         111   11340       3
% 66      11Tumor                             174   12533      11
% 67      SMK-CAN-187                         187   19993       2
% 68      GLI-85                               85   22283       2

    properties(Access = private)
        TrainIn;    % Input of training set
        TrainOut;   % Output of training set
        TestIn;    % Input of validation set
        TestOut;   % Output of validation set
        Category;   % Output label set
        indices;
        dataNo;
        TestNum;
        K;          % The number of nearest neighbor in KNN
    end

    properties(Access = public)
        boolTrain; % 1 means apply the data to the training set, otherwise to the test set
        TrainX;
        TrainY;
    end

    methods
        %% Default settings of the problem
        function Setting(obj)
            [dataNo, encode, objectiveNo, NoiseLevel] = obj.ParameterSet(1, 1, 2, 0);
            addpath(genpath('FSdata'));
            %% Load data
            str = {'Glass.mat',      'Wine.mat',          'Leaf.mat',              'Australian.mat',       'Zoo.mat',...
                'Lymph.mat',            'Vehicle.mat',       'ImageSegmentation.mat', 'Parkinson.mat',        'Spect.mat',...
                'German.mat',           'LedDisplay.mat',    'WallRobot.mat',         'WBCD.mat',             'GuesterPhase.mat',...
                'Dermatology.mat',      'Ionosphere.mat',    'Chess.mat',             'Connect4.mat',         'Lung.mat',...
                'Sonar.mat',            'Plant.mat',         'Mice.mat',              'Movementlibras.mat',   'Hillvalley.mat',...
                'GasSensor.mat',        'Musk1.mat',         'USPS.mat',              'Semeion.mat',          'Arrhythmia.mat',...
                'LSVT.mat',             'lung_discrete.mat', 'Madelon.mat',           'Isolet.mat',           'Isolet5.mat',...
                'MultipleFeatures.mat', 'Gametes.mat',       'ORL.mat',               'Yale.mat',             'COIL20.mat',...
                'Christine.mat',        'Bioresponse.mat',   'colon.mat',             'Colon.mat',            'SRBCT.mat',...
                'AR10P.mat',            'lymphoma.mat',      'RELATHE.mat',           'GLIOMA.mat',           'BASEHOCK.mat',...
                'Gisette.mat',          'Leukemia1',         'DLBCL.mat',             '9Tumor.mat',           'Brain1.mat',...
                'Prostate-GE.mat',      'CNS.mat',           'Leukemia.mat',          'ALLAML.mat',           'Carcinom.mat',...
                'nci9.mat',             'arcene.mat',        'orlraws10P.mat',        'Brain2.mat',           'CLL-SUB-111.mat',...
                '11Tumor.mat',          'SMK-CAN-187.mat',   'GLI-85.mat'};
            
            Data = load(str{dataNo});
            obj.dataNo = dataNo;
            obj.boolTrain = 1;
            
            %% Shuffle the order of instances
            rng(1);   %Fix the random number seed
            randIndex = randperm(size(Data.X, 1));
            Data.X = Data.X(randIndex',:);
            Data.Y = Data.Y(randIndex',:);


            %% Add noise to each feature
            noiseIndex = randperm(size(Data.X, 2), round(NoiseLevel./100.*size(Data.X, 2)));
            noiseVal = rand(size(Data.X, 1), round(NoiseLevel./100.*size(Data.X, 2)));
            Data.X(1:size(Data.X, 1)*0.7, noiseIndex) = noiseVal(1:size(Data.X, 1)*0.7, :);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %% Normalize the input data
            Fmin         = min(Data.X, [], 1);
            Fmax         = max(Data.X, [], 1);
            Data.X       = (Data.X - repmat(Fmin, size(Data.X, 1), 1))./repmat(Fmax - Fmin, size(Data.X, 1), 1);
            obj.Category = unique(Data.Y);
            %% Divide the training set and test set
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.TrainX = Data.X(1:ceil(end*0.7), 1:end);
            obj.TrainY = Data.Y(1:ceil(end*0.7), 1:end);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.TrainIn  = Data.X(1:ceil(end*0.7), 1:end);
            obj.TrainOut = Data.Y(1:ceil(end*0.7), 1:end);
            obj.TestIn   = Data.X(ceil(end*0.7)+1:end, 1:end);
            obj.TestOut  = Data.Y(ceil(end*0.7)+1:end, 1:end);
            obj.TestNum  = 5;   % Produce 5 folds 
            obj.indices  = crossvalind('Kfold', size(obj.TrainIn,1), obj.TestNum);     % Produce 5 folds 
            obj.K        = 5;   % The number of nearest neighbor in KNN
            % Number of objectives and features
            obj.M        = objectiveNo;
            obj.D        = size(obj.TrainIn, 2);
            switch encode
                case 1
                    obj.encoding = 'binary';
                case 2
                    obj.encoding = 'real';
                    obj.lower    = zeros(1, obj.D);
                    obj.upper    = ones( 1, obj.D);
            end
        end
        %% Calculate objective values
        function PopObj = CalObj(obj, PopDec)
            PopObj  = zeros(size(PopDec, 1), obj.M);
            if obj.boolTrain == 1   % To judge whether in the training set or the test set
                %%%%% For training set %%%%%
                theta = 0.5;     % The threshold to determine whether a feature is selected or discarded
                PopDec(PopDec < theta)  = 0 ;
                PopDec(PopDec >= theta) = 1;
                PopDec = logical(PopDec);                 
                for i = 1 : size(PopObj,1)
                    sumAccuracyRatio = 0;
                    for j = 1:obj.TestNum                  % k-fold cross validation
                        test     = (obj.indices == j);           % Every iteration selects a fold as the sub test set 
                        train    =~ test;                
                        TrainInsub  = obj.TrainIn(train, :);  
                        TrainOutsub = obj.TrainOut(train, :); 
                        TestInsub  = obj.TrainIn(test, :); 
                        TestOutsub = obj.TrainOut(test, :);
                        [~, Rank] = sort(pdist2(TestInsub(:, PopDec(i, :)), TrainInsub(:, PopDec(i, :))), 2);
                        [~, Out]  = max(hist(double(TrainOutsub(Rank(:, 1:obj.K))'), double(obj.Category)), [], 1);
                        Out       = obj.Category(Out);
                        BalanceAccuracy = 0;
                        for t = 0:size(obj.Category, 1)-1
                            index = Out==t;
                            if sum(index) > 0
                                BalanceAccuracy = BalanceAccuracy + mean(Out(index)==TestOutsub(index));
                            end
                        end
                        sumAccuracyRatio   = sumAccuracyRatio + BalanceAccuracy./size(obj.Category, 1);
                    end
                    AccuracyRatio = sumAccuracyRatio./obj.TestNum;
                    errorRatio    = 1 - AccuracyRatio;
                    switch obj.M
                        case 1    % Single-objective feature selection
                            alpha = 1e-6;
                            PopObj(i, 1) = errorRatio.*(1-alpha) + mean(PopDec(i, :)).*alpha;
                        case 2    % Bi-objective feature selection
                            PopObj(i, 1) = errorRatio;
                            PopObj(i, 2) = mean(PopDec(i, :));
                    end
                end 
            else
                %%%%% For test set %%%%%
                PopDec = logical(PopDec);
                for i = 1 : size(PopObj, 1)
                    [~, Rank] = sort(pdist2(obj.TestIn(:, PopDec(i, :)), obj.TrainIn(:, PopDec(i, :))), 2);
                    [~, Out]  = max(hist(double(obj.TrainOut(Rank(:, 1:obj.K)))', double(obj.Category)), [], 1);
                    Out       = obj.Category(Out);
                    BalancedAccuracy = 0;
                    for t = 0:size(obj.Category, 1)-1
                        index = Out==t;
                        if sum(index) > 0
                            BalancedAccuracy = BalancedAccuracy + mean(Out(index)==obj.TestOut(index));
                        end
                    end
                    BalancedError = 1 - BalancedAccuracy./size(obj.Category, 1);
                    if obj.M ==1
                        PopObj(i, 1) = -BalancedAccuracy./size(obj.Category, 1);  % Maximize the classification accuracy
                    elseif obj.M == 2
                        PopObj(i, 1) = BalancedError;
                        PopObj(i, 2) = mean(PopDec(i, :));
                    else
                        disp('Input error: No such setting!');
                    end
                end 
            end
        end
        
        %% Calculate constraints
%         function PopCon = CalCon(obj, PopDec)
%             PopObj      = obj.CalObj(PopDec);
%             ind         = ones(1, obj.D);   % The feature subset that select all features
%             [~, Rank]   = sort(pdist2(obj.TrainIn(:, ind), obj.TrainIn(:, ind)), 2);
%             [~, Out]    = max(hist(double(obj.TrainOut(Rank(:, 1:obj.K))'), double(obj.Category)), [], 1);
%             Acc         = mean(obj.Category(Out)==obj.TrainOut);
%             Err         = 1 - Acc;
%             PopCon(:,1) = PopObj(:,1) - Err; % The classification error should smaller than that of using all features
%         end
        
        %% Generate points on the Pareto front
%         function R = GetOptimum(obj, N)
%             %addpath(genpath('scripts'));
%             dataNo = obj.dataNo;
%             fullfile = ['FS_', num2str(dataNo), '.mat'];
%             P = load(fullfile);
%             R = P.NDpoints;
%         end 
        %% Display a population in the objective space
        function DrawObj(obj, Population)
            Draw(Population.objs, {'Classification error rate', 'Selected feature ratio', []});
        end 
    end
end
