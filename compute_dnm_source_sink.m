%% 输入：EEG data
% eegData: N_channels × T_timepoints
% fs: 采样率（如1000Hz）
% window_ms: 窗口长度（如125ms）

function DNM_results = compute_dnm_source_sink(eegData, fs, window_ms)

    %% 参数准备
    [N, T] = size(eegData);
    window_size = round(fs * window_ms / 1000);   % 每窗多少采样点
    num_windows = floor(T / window_size);
    
    A_list = cell(1, num_windows);  % 存储每个时间窗的 A 矩阵
    source_all = zeros(N, num_windows);
    sink_all = zeros(N, num_windows);
    source_rank_all = zeros(N, num_windows);
    sink_rank_all = zeros(N, num_windows);

    %% 主循环：每个时间窗估计 A，并计算 source/sink
    for w = 1:num_windows
        idx_start = (w - 1) * window_size + 1;
        idx_end = idx_start + window_size - 1;
        segment = eegData(:, idx_start:idx_end);  % N × W

        % 构建输入X和输出Y矩阵
        X = segment(:, 1:end-1);  % N × (W-1)
        Y = segment(:, 2:end);    % N × (W-1)

        % 最小二乘解：Y = A * X → A = Y * pinv(X)
        A = Y * pinv(X);  % A为 N×N
        A_list{w} = A;

        % 计算列范数（source）和行范数（sink）
        source_vec = vecnorm(A, 2, 1)';  % N×1，每列的L2范数
        sink_vec = vecnorm(A, 2, 2);     % N×1，每行的L2范数

        % 排名（越大表示越像source/sink）
        [~, sr_idx] = sort(source_vec, 'ascend');
        source_rank = zeros(N, 1);
        source_rank(sr_idx) = 1:N;

        [~, sk_idx] = sort(sink_vec, 'descend');
        sink_rank = zeros(N, 1);
        sink_rank(sk_idx) = 1:N;

        % 存储
        source_all(:, w) = source_vec;
        sink_all(:, w) = sink_vec;
        source_rank_all(:, w) = source_rank;
        sink_rank_all(:, w) = sink_rank;
    end

    %% 输出结构体
    DNM_results.A_list = A_list;
    DNM_results.source = source_all;
    DNM_results.sink = sink_all;
    DNM_results.source_rank = source_rank_all;
    DNM_results.sink_rank = sink_rank_all;

end
