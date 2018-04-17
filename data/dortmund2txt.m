datasets = strvcat('Synthie');

for ith_data = 1: size(datasets, 1)
    dataset = strcat(datasets(ith_data, :));
    path2data = [dataset '/'];    % path to folder containing data
    
    % graph labels
    graph_labels = dlmread([path2data dataset '_graph_labels.txt']);
    num_graphs = size(graph_labels, 1);
    
    graph_ind = dlmread([path2data dataset '_graph_indicator.txt']);
    num_nodes = size(graph_ind, 1);
    
    % node labels
    try
        labels = dlmread([path2data dataset '_node_labels.txt']);
        if size(labels,1) ~= num_nodes
            fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_labels.txt']);
        end
    catch
        disp('No node labels for this dataset.')
    end
    
    % node attributes
    try
        attributes = dlmread([path2data dataset '_node_attributes.txt']);
        if size(attributes,1) ~= num_nodes
            fprintf('ERROR: Wrong number of nodes in %s!\n', [dataset '_node_attributes.txt']);
        end
    catch
        disp('No node attributes for this dataset.')
    end
    
    % edges, adjacency matrix
    edges = dlmread([path2data dataset '_A.txt']);
    num_edges = size(edges,1);
    
    % edge attributes (e.g. edge weights etc.)
    try
        edge_attr = dlmread([path2data dataset '_edge_attributes.txt']);
        if size(edge_attr, 1) ~= num_edges
            fprintf('ERROR: Wrong number of edges in %s!\n', [dataset '_edge_attributes.txt']);
        end
        if size(edge_attr,2) > 1
            fprintf('CAUTION: there are more than one edge attributes in %s!\n', [dataset '_edge_attributes.txt']);
            fprintf('CAUTION: only the first one is used in adjacency matrix.\n');
        end
    catch
        edge_attr = ones(num_edges, 1);
        disp('No edge attributes for this dataset.')
    end
    
    A = spones(sparse(edges(:,1), edges(:,2), edge_attr(:,1), num_nodes, num_nodes));

    % save to a single txt 
    graph_labels = graph_labels - min(graph_labels);
    fid = fopen(sprintf('%s/%s.txt', dataset, dataset), 'w');
    fprintf(fid, '%d\n', num_graphs);

    for i = 1 : num_graphs
        i_ind = graph_ind == i;
        Ai = A(i_ind, i_ind);
        if exist('labels')
            li = labels(i_ind);
        end
        if exist('attributes')
            attri = attributes(i_ind, :);
        end
        start_ind = nnz(graph_ind < i);
        num_nodes_i = size(Ai, 1);
        fprintf(fid, '%d %d\n', num_nodes_i, graph_labels(i));
        for j = 1 : num_nodes_i
            rowj = Ai(j, :);
            neighbors = find(rowj);
            num_neighbors = length(neighbors); 
            if ~exist('labels')
                fprintf(fid, '%d %d', 0, num_neighbors);  % give a null label
            else
                fprintf(fid, '%d %d', li(j) - 1, num_neighbors);  % node label starts from 0
            end
            for k = 1 : num_neighbors
                fprintf(fid, ' %d', neighbors(k) - 1);
            end
            if exist('attributes')
                for k = 1 : length(attri(j, :))
                    fprintf(fid, ' %.6f', attri(j, k));
                end
            end
            fprintf(fid, '\n');
        end
    end

    fclose(fid);

    total = length(graph_labels);
    fold_size = floor(total / 10);
    p = randperm(total);
    mkdir(dataset, '10fold_idx')
    for fold = 1 : 10
        test_range = (fold - 1) * fold_size + 1 : fold * fold_size;
        train_range = [1 : (fold - 1) * fold_size, fold * fold_size + 1 : total];
        
        fid = fopen(sprintf('%s/10fold_idx/test_idx-%d.txt', dataset, fold), 'w');
        for i = 1 : length(test_range)
            fprintf(fid, '%d\n', p(test_range(i)) - 1);
        end
        fclose(fid);

        fid = fopen(sprintf('%s/10fold_idx/train_idx-%d.txt', dataset, fold), 'w');
        for i = 1 : length(train_range)
            fprintf(fid, '%d\n', p(train_range(i)) - 1);
        end
        fclose(fid);
    end
    
end
