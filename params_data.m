function [paras_data] = params_data(dataName)
%
% (c) Ivica Kopriva, January 2025
%
% This function returns parameters for chosen dataset in structure paras_data,
%

if strcmp(dataName,'YaleBCrop025')
    % Dataset dependent parameters
    paras_data.i1 = 48; paras_data.i2 = 42; % image size
    paras_data.dimSubspace=9; % % subspaces dimension
    paras_data.numIn = 43; % number of in-sample data
    paras_data.numOut = 21; % number of out-of-sample data
    paras_data.nc = 38; % number of groups

    load YaleBCrop025.mat;
    [i1, i2, ni, nc] = size(I); % rectangular image size: i1 x i2, number of images per person: ni, number of persons: nc
    clear I Ind s ns

    N = nc*ni; % number of samples
    X = zeros(i1*i2, N); labels = zeros(N,1); % allocation of space

    ns = 0; % sample number counter
    for i=1:nc % person
        for j=1:ni % face image
            ns = ns + 1; % sample index
            X(:,ns) = Y(:,j,i); % sample (columns of X represent vectorized data of rectangular images)
            labels(ns,1) = i;    % to be used for oracle based validation
        end
    end
    paras_data.X=X;
    paras_data.labels=labels;

elseif strcmp(dataName,'MNIST')
    % Dataset dependant parameters
    paras_data.i1 = 28; paras_data.i2 = 28; % image size
    paras_data.dimSubspace=12; % % subspaces dimension
    paras_data.numIn = 50; % number of in-sample data
    paras_data.numOut = 50; % number of out-of-sample data
    paras_data.nc = 10; % number of groups

    X = loadMNISTImages('t10k-images.idx3-ubyte'); % columns of X represent vectorized data of squared images
    labels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % to be used for oracle based validation
    [labelssorted,IX] = sort(labels);

    labels = labelssorted + 1;
    paras_data.X=X(:,IX);
    paras_data.labels=labels;

elseif strcmp(dataName,'USPS')
    % Dataset dependant parameters
    paras_data.i1 = 16; paras_data.i2 = 16; % image size
    paras_data.dimSubspace=12; % % subspaces dimension
    paras_data.numIn = 50; % number of in-sample data
    paras_data.numOut = 50; % number of out-of-sample data
    paras_data.nc = 10; % number of groups

    data = load('usps');
    Y = data(:,2:end)'; % columns of Y represent vectorized data of squared images
    labels = data(:,1);
    clear data

    paras_data.X=Y;
    paras_data.labels=labels;

elseif strcmp(dataName,'ORL')
    % Dataset dependent parameters
    paras_data.i1 = 32; paras_data.i2 = 32; % image size
    paras_data.dimSubspace=5; % % subspaces dimension
    paras_data.numIn = 7; % number of in-sample data
    paras_data.numOut = 3; % number of out-of-sample data
    paras_data.nc = 40; % number of groups

    data = load('ORL_32x32.mat');
    Y = data.fea'; % columns of X represent vectorized data of squared images
    labels=data.gnd;

    paras_data.X=Y;
    paras_data.labels=labels;

elseif strcmp(dataName,'COIL20')
    % Dataset dependant parameters
    paras_data.i1 = 32; paras_data.i2 = 32; % image size
    paras_data.dimSubspace=10; % % subspaces dimension
    paras_data.numIn = 50; % number of in-sample data
    paras_data.numOut = 22; % number of out-of-sample data
    paras_data.nc = 20; % number of groups

    load COIL20.mat
    X=transpose(fea); % columns of X represent vectorized data of squared images
    % i1=32; i2=32; N=1440; nc=20; % 1440 images of 20 objects (72 images per object) (each image is 32x32 pixels)
    clear fea;

    labels=gnd;   % to be used for oracle based validation
    % nc = 20; % twenty objects images
    clear gnd

    paras_data.X=X;
    paras_data.labels=labels;

elseif strcmp(dataName,'COIL100')
    % Dataset dependant parameters
    paras_data.i1=32; paras_data.i2=32; % image dimensions
    paras_data.dimSubspace=10;   % subspaces dimension
    paras_data.numIn = 50;  % number of in-sample data
    paras_data.numOut = 22; % number of out-of-sample data
    paras_data.nc = 100; % number of groups

    load COIL100.mat
    X=double(fea.'); % columns of X represent vectorized data of squared images
    clear fea;

    labels=gnd;    % to be used for oracle based validation
    clear gnd

    paras_data.X=X;
    paras_data.labels=labels;

end

end