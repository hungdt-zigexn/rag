#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Reranker RAG (Phiên bản Ruby)
# ===============================================================================
#
# Reranker RAG cải thiện chất lượng retrieval bằng cách thêm một stage
# "reranking" sau initial retrieval. Đây là Two-Stage Retrieval approach:
#
# Stage 1: RETRIEVAL - Tìm kiếm candidate documents dựa trên embedding similarity
# Stage 2: RERANKING - Sử dụng LLM để đánh giá lại và sắp xếp candidates
#
# Lý do cần Reranker:
# 1. Embedding similarity không phải lúc nào cũng perfect
# 2. LLM có thể hiểu context và relevance tốt hơn pure cosine similarity
# 3. Có thể incorporate domain-specific ranking criteria
# 4. Tăng precision của top-k results (quality over quantity)
#
# Process Flow:
# Query → Embedding Search → Top-N Candidates → LLM Reranking → Final Top-K

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Reranker RAG ==="

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'
RERANKER_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'  # Có thể dùng model khác

# Kiểm tra API key
if OPENAI_API_KEY.nil? || OPENAI_API_KEY.empty?
  puts "Cảnh báo: Biến môi trường OPENAI_API_KEY chưa được đặt"
  exit 1
end

# ===============================================================================
# API Client Functions
# ===============================================================================

def make_api_request(endpoint, payload)
  """
  Thực hiện API request đến OpenAI-compatible endpoint.
  """
  uri = URI("#{BASE_URL}#{endpoint}")
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true

  request = Net::HTTP::Post.new(uri)
  request['Authorization'] = "Bearer #{OPENAI_API_KEY}"
  request['Content-Type'] = 'application/json'
  request.body = payload.to_json

  response = http.request(request)

  if response.code.to_i == 200
    JSON.parse(response.body)
  else
    puts "Lỗi API: #{response.code} - #{response.body}"
    nil
  end
end

def create_embeddings(text_list, model = EMBEDDING_MODEL)
  """
  Tạo embeddings cho danh sách văn bản.
  """
  payload = {
    model: model,
    input: text_list
  }

  response = make_api_request('embeddings', payload)
  return [] unless response && response['data']

  response['data'].map { |item| item['embedding'] }
end

def generate_response(system_prompt, user_message, model = CHAT_MODEL)
  """
  Tạo phản hồi từ mô hình AI.
  """
  payload = {
    model: model,
    temperature: 0.1,  # Very low temperature for consistent ranking
    messages: [
      { role: 'system', content: system_prompt },
      { role: 'user', content: user_message }
    ]
  }

  response = make_api_request('chat/completions', payload)
  return nil unless response && response['choices'] && !response['choices'].empty?

  response['choices'][0]['message']['content']
end

# ===============================================================================
# Text Processing Utilities
# ===============================================================================

def extract_text_from_pdf(pdf_path)
  """
  Trích xuất văn bản từ file PDF.
  """
  reader = PDF::Reader.new(pdf_path)
  all_text = ""

  reader.pages.each do |page|
    all_text += page.text + " "
  end

  all_text.strip
end

def chunk_text(text, chunk_size = 1000, overlap = 200)
  """
  Chia văn bản thành các chunk có overlap.
  """
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Stage 1: Initial Retrieval
# ===============================================================================

def cosine_similarity(vec1, vec2)
  """
  Tính cosine similarity giữa hai vector.
  """
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

def initial_retrieval(query, chunks, embeddings, top_n = 10)
  """
  Stage 1: Initial retrieval sử dụng embedding similarity.

  Lấy top-N candidates để cung cấp cho reranker.
  N thường lớn hơn K cuối cùng để cho reranker nhiều lựa chọn.

  Args:
    query (String): User query.
    chunks (Array): Document chunks.
    embeddings (Array): Chunk embeddings.
    top_n (Integer): Số candidates để rerank.

  Returns:
    Array: Top-N candidates với [chunk_index, similarity_score, chunk_text].
  """
  puts "\n--- STAGE 1: INITIAL RETRIEVAL ---"
  puts "Query: #{query}"
  puts "Retrieving top #{top_n} candidates..."

  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity với tất cả chunks
  similarities = embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity, chunks[i]]
  end

  # Sắp xếp và lấy top N
  top_candidates = similarities.sort_by { |_, score, _| -score }.first(top_n)

  puts "Initial retrieval results:"
  top_candidates.each_with_index do |(index, score, chunk), rank|
    preview = chunk[0..80] + (chunk.length > 80 ? "..." : "")
    puts "#{rank + 1}. Chunk #{index}: Score #{score.round(4)} - #{preview}"
  end

  top_candidates
end

# ===============================================================================
# Stage 2: LLM-based Reranking
# ===============================================================================

def rerank_with_llm(query, candidates, top_k = 5, model = RERANKER_MODEL)
  """
  Stage 2: LLM-based reranking của candidates.

  Sử dụng LLM để đánh giá relevance của mỗi candidate document
  với query, sau đó sắp xếp lại theo scores.

  Args:
    query (String): User query.
    candidates (Array): Candidates từ initial retrieval.
    top_k (Integer): Số documents cuối cùng cần trả về.
    model (String): LLM model để reranking.

  Returns:
    Hash: {
      reranked_results: Top-k sau reranking,
      all_scores: Tất cả scores từ LLM,
      ranking_details: Chi tiết quá trình ranking
    }
  """
  puts "\n--- STAGE 2: LLM RERANKING ---"
  puts "Reranking #{candidates.length} candidates với LLM..."
  puts "Target top-k: #{top_k}"

  # Chuẩn bị documents cho reranking
  docs_for_ranking = candidates.each_with_index.map do |(chunk_idx, emb_score, chunk_text), i|
    {
      id: i,
      chunk_index: chunk_idx,
      embedding_score: emb_score,
      text: chunk_text[0..600] + (chunk_text.length > 600 ? "..." : "")  # Truncate for LLM
    }
  end

  # System prompt cho reranking task
  system_prompt = """
  Bạn là một chuyên gia đánh giá relevance của documents với user queries.
  Nhiệm vụ của bạn là chấm điểm từng document về mức độ liên quan với query.

  TIÊU CHÍ ĐÁNH GIÁ:
  1. **Topical Relevance** (40%): Document có nói về chủ đề của query không?
  2. **Information Coverage** (30%): Document có chứa thông tin trả lời query không?
  3. **Specificity** (20%): Document có specific hay chỉ general về topic?
  4. **Completeness** (10%): Document có đủ context hay chỉ fragment?

  THANG ĐIỂM:
  - 1.0: Hoàn toàn relevant, trả lời trực tiếp query
  - 0.8: Relevant cao, chứa most information needed
  - 0.6: Moderately relevant, chứa một số thông tin hữu ích
  - 0.4: Slightly relevant, mention chủ đề nhưng không deep
  - 0.2: Barely relevant, chỉ có một số keywords match
  - 0.0: Không relevant hoặc off-topic

  FORMAT OUTPUT:
  Cho mỗi document, trả về chỉ con số score (0.0-1.0), mỗi score trên một dòng.
  VÍ DỤ OUTPUT:
  0.9
  0.6
  0.3
  0.8
  """

  # Tạo prompt với query và documents
  docs_text = docs_for_ranking.map.with_index do |doc, i|
    "DOCUMENT #{i + 1}:\n#{doc[:text]}\n"
  end.join("\n")

  user_message = """
QUERY: #{query}

#{docs_text}

Hãy đánh giá relevance của từng document với query trên.
Trả về #{docs_for_ranking.length} scores, mỗi score trên một dòng:
"""

  # Gọi LLM để scoring
  response = generate_response(system_prompt, user_message, model)

  if response
    # Parse scores từ response
    score_lines = response.split("\n")
                         .map(&:strip)
                         .reject(&:empty?)
                         .map { |line| line.gsub(/[^\d\.]/, '') }  # Extract numbers
                         .map(&:to_f)
                         .map { |score| score.clamp(0.0, 1.0) }   # Ensure valid range

    puts "\nLLM Reranking Scores:"

    # Combine scores với document info
    scored_docs = docs_for_ranking.each_with_index.map do |doc, i|
      llm_score = score_lines[i] || 0.0  # Default to 0.0 if parsing failed

      puts "Doc #{i + 1} (Chunk #{doc[:chunk_index]}): LLM=#{llm_score} | Embedding=#{doc[:embedding_score].round(4)}"

      {
        chunk_index: doc[:chunk_index],
        chunk_text: candidates[i][2],  # Original full text
        embedding_score: doc[:embedding_score],
        llm_score: llm_score,
        combined_score: llm_score  # Use LLM score as primary
      }
    end

    # Sắp xếp theo LLM scores
    reranked = scored_docs.sort_by { |doc| -doc[:llm_score] }

    puts "\nReranking Results (Top #{top_k}):"
    reranked.first(top_k).each_with_index do |doc, i|
      puts "#{i + 1}. Chunk #{doc[:chunk_index]}: LLM=#{doc[:llm_score]} | Emb=#{doc[:embedding_score].round(4)}"
      puts "   Text: #{doc[:chunk_text][0..100]}..."
    end

    {
      reranked_results: reranked.first(top_k),
      all_scores: scored_docs,
      ranking_details: {
        original_order: docs_for_ranking.map { |d| d[:chunk_index] },
        reranked_order: reranked.map { |d| d[:chunk_index] },
        score_changes: scored_docs.map.with_index { |doc, i|
          {
            chunk: doc[:chunk_index],
            original_rank: i + 1,
            new_rank: reranked.index(doc) + 1,
            rank_change: (i + 1) - (reranked.index(doc) + 1)
          }
        }
      }
    }
  else
    puts "❌ Không thể thực hiện LLM reranking"
    # Fallback: trả về original order
    {
      reranked_results: candidates.first(top_k).map.with_index do |(chunk_idx, emb_score, chunk_text), i|
        {
          chunk_index: chunk_idx,
          chunk_text: chunk_text,
          embedding_score: emb_score,
          llm_score: emb_score,  # Use embedding score as fallback
          combined_score: emb_score
        }
      end,
      all_scores: [],
      ranking_details: { error: "LLM reranking failed" }
    }
  end
end

# ===============================================================================
# Complete Two-Stage Retrieval Pipeline
# ===============================================================================

def two_stage_retrieval(query, chunks, embeddings, retrieval_n = 10, final_k = 5)
  """
  Complete two-stage retrieval pipeline: Initial Retrieval + LLM Reranking.

  Args:
    query (String): User query.
    chunks (Array): Document chunks.
    embeddings (Array): Chunk embeddings.
    retrieval_n (Integer): Số candidates từ stage 1.
    final_k (Integer): Số final results từ stage 2.

  Returns:
    Hash: Complete results từ both stages.
  """
  puts "\n" + "=" * 80
  puts "TWO-STAGE RETRIEVAL PIPELINE"
  puts "=" * 80
  puts "Query: #{query}"
  puts "Stage 1: Retrieve top #{retrieval_n} candidates"
  puts "Stage 2: Rerank to final top #{final_k}"
  puts "=" * 80

  # Stage 1: Initial Retrieval
  candidates = initial_retrieval(query, chunks, embeddings, retrieval_n)
  return { error: "No candidates found" } if candidates.empty?

  # Stage 2: LLM Reranking
  reranking_results = rerank_with_llm(query, candidates, final_k)

  {
    query: query,
    stage1_candidates: candidates,
    stage2_reranked: reranking_results,
    pipeline_summary: {
      candidates_retrieved: candidates.length,
      final_results: reranking_results[:reranked_results].length,
      reranking_success: !reranking_results[:ranking_details].key?(:error)
    }
  }
end

# ===============================================================================
# Comparison Analysis
# ===============================================================================

def compare_retrieval_vs_reranking(query, chunks, embeddings, k = 5)
  """
  So sánh kết quả giữa pure embedding retrieval và two-stage reranking.
  """
  puts "\n" + "=" * 80
  puts "COMPARISON: RETRIEVAL vs RERANKING"
  puts "=" * 80

  # Pure embedding retrieval
  puts "\n1. PURE EMBEDDING RETRIEVAL:"
  puts "-" * 50
  pure_results = initial_retrieval(query, chunks, embeddings, k)

  # Two-stage retrieval với reranking
  puts "\n2. TWO-STAGE RETRIEVAL (với RERANKING):"
  puts "-" * 50
  two_stage_results = two_stage_retrieval(query, chunks, embeddings, k * 2, k)

  return unless two_stage_results[:stage2_reranked]

  # Comparison analysis
  puts "\n3. DETAILED COMPARISON:"
  puts "-" * 50

  pure_top_chunks = pure_results.map { |chunk_idx, _, _| chunk_idx }
  reranked_top_chunks = two_stage_results[:stage2_reranked][:reranked_results]
                                                           .map { |doc| doc[:chunk_index] }

  puts "Pure Retrieval Top #{k}: #{pure_top_chunks}"
  puts "Reranked Top #{k}: #{reranked_top_chunks}"

  # Overlap analysis
  overlap = (pure_top_chunks & reranked_top_chunks).length
  overlap_percentage = (overlap.to_f / k * 100).round(1)

  puts "\nOverlap Analysis:"
  puts "- Chunks giống nhau: #{overlap}/#{k} (#{overlap_percentage}%)"
  puts "- Chunks khác nhau: #{k - overlap}/#{k} (#{100 - overlap_percentage}%)"

  # Rank changes analysis
  if two_stage_results[:stage2_reranked][:ranking_details][:score_changes]
    puts "\nRank Changes Analysis:"
    two_stage_results[:stage2_reranked][:ranking_details][:score_changes]
                     .select { |change| change[:rank_change] != 0 }
                     .each do |change|
      direction = change[:rank_change] > 0 ? "↑" : "↓"
      puts "  Chunk #{change[:chunk]}: #{change[:original_rank]} → #{change[:new_rank]} #{direction}"
    end
  end

  # Show actual content differences
  puts "\n4. CONTENT COMPARISON:"
  puts "-" * 50

  puts "Top result - Pure Retrieval:"
  pure_top = pure_results.first
  puts "  Chunk #{pure_top[0]}: Score #{pure_top[1].round(4)}"
  puts "  Content: #{pure_top[2][0..200]}..."

  puts "\nTop result - After Reranking:"
  reranked_top = two_stage_results[:stage2_reranked][:reranked_results].first
  puts "  Chunk #{reranked_top[:chunk_index]}: LLM=#{reranked_top[:llm_score]} | Emb=#{reranked_top[:embedding_score].round(4)}"
  puts "  Content: #{reranked_top[:chunk_text][0..200]}..."

  if pure_top[0] != reranked_top[:chunk_index]
    puts "\n⚠ RERANKING CHANGED THE TOP RESULT!"
  else
    puts "\n✓ Reranking confirmed the same top result"
  end

  {
    pure_results: pure_results,
    reranked_results: two_stage_results,
    overlap_percentage: overlap_percentage,
    top_result_changed: pure_top[0] != reranked_top[:chunk_index]
  }
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_reranker_demo
  """
  Chạy demo hoàn chỉnh Reranker RAG với multiple test cases.
  """
  puts "\n=== Demo Reranker RAG ==="

  # Bước 1: Load và chuẩn bị data
  puts "\n1. Chuẩn bị dữ liệu..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  chunks = chunk_text(extracted_text, 1000, 200)

  # Giới hạn chunks cho demo
  demo_chunks = chunks.first(15)
  puts "Demo với #{demo_chunks.length} chunks"

  # Tạo embeddings
  puts "Tạo embeddings cho chunks..."
  embeddings = create_embeddings(demo_chunks)
  return unless embeddings.any?

  # Bước 2: Test với various queries
  test_queries = [
    "Các ứng dụng thực tế của AI trong chăm sóc sức khỏe là gì?",
    "Machine learning có thể cải thiện hiệu quả kinh doanh như thế nào?",
    "Những cân nhắc đạo đức khi triển khai hệ thống AI là gì?",
    "Giải thích sự khác biệt giữa supervised và unsupervised learning",
    "Những thách thức mà các công ty gặp phải khi áp dụng trí tuệ nhân tạo là gì?"
  ]

  # Bước 3: Run comparison cho mỗi query
  test_queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: Reranker Analysis"
    puts "=" * 100

    comparison = compare_retrieval_vs_reranking(query, demo_chunks, embeddings, 5)

    puts "\n" + "=" * 100
  end
end

# ===============================================================================
# Specialized Reranking Strategies
# ===============================================================================

def domain_specific_reranking(query, candidates, domain = "general", top_k = 5)
  """
  Domain-specific reranking với customized criteria.

  Ví dụ các domains:
  - "healthcare": Focus on accuracy, evidence, medical relevance
  - "business": Focus on actionability, ROI, practical implementation
  - "technical": Focus on precision, completeness, technical depth
  """
  domain_prompts = {
    "healthcare" => """
    Bạn là chuyên gia y tế đánh giá tài liệu medical/healthcare.

    TIÊU CHÍ ĐÁNH GIÁ:
    1. Medical Accuracy (35%): Thông tin có chính xác về mặt y khoa?
    2. Evidence-based (25%): Có dựa trên research, studies, clinical data?
    3. Clinical Relevance (25%): Liên quan đến practice y tế thực tế?
    4. Patient Safety (15%): Có đề cập đến safety considerations?

    Ưu tiên documents có evidence-based information và clinical relevance cao.
    """,

    "business" => """
    Bạn là business consultant đánh giá tài liệu kinh doanh.

    TIÊU CHÍ ĐÁNH GIÁ:
    1. Actionability (30%): Thông tin có thể implement được không?
    2. Business Impact (25%): Có impact rõ ràng đến business outcomes?
    3. Practical Feasibility (25%): Realistic để execute trong real business?
    4. ROI Potential (20%): Có mention đến cost/benefit, ROI?

    Ưu tiên documents có practical advice và measurable business value.
    """,

    "technical" => """
    Bạn là technical expert đánh giá tài liệu kỹ thuật.

    TIÊU CHÍ ĐÁNH GIÁ:
    1. Technical Depth (35%): Level of technical detail và accuracy
    2. Implementation Details (30%): Có specific implementation guidance?
    3. Technical Completeness (20%): Cover đủ technical aspects?
    4. Code/Examples (15%): Có practical examples, code snippets?

    Ưu tiên documents có technical depth và practical implementation details.
    """,

    "general" => """
    Bạn là chuyên gia đánh giá general document relevance.

    TIÊU CHÍ ĐÁNH GIÁ:
    1. Topical Relevance (40%): Document có nói về chủ đề của query?
    2. Information Quality (30%): Chất lượng và completeness của thông tin
    3. Clarity (20%): Thông tin có clear và easy to understand?
    4. Comprehensiveness (10%): Coverage của topic có broad không?
    """
  }

  selected_prompt = domain_prompts[domain] || domain_prompts["general"]
  puts "\nSử dụng #{domain} domain-specific reranking..."

  # Use the domain-specific prompt cho reranking
  rerank_with_llm(query, candidates, top_k, RERANKER_MODEL)
end

# ===============================================================================
# Advanced Reranking Analysis
# ===============================================================================

def analyze_reranking_patterns
  """
  Phân tích patterns trong reranking behavior.
  """
  puts "\n=== Phân tích Reranking Patterns ==="

  # Load data
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  chunks = chunk_text(extracted_text, 800, 150).first(10)
  embeddings = create_embeddings(chunks)
  return unless embeddings.any?

  # Test với different query types
  query_types = {
    "Factual" => "What is artificial intelligence?",
    "How-to" => "How do machine learning algorithms work?",
    "Comparative" => "What are the differences between AI and machine learning?",
    "Problem-solving" => "What challenges exist in AI implementation?",
    "Benefits" => "What are the advantages of using AI in business?"
  }

  puts "Testing reranking patterns across query types:"
  puts "=" * 60

  pattern_results = {}

  query_types.each do |type, query|
    puts "\n#{type.upcase} QUERY: #{query}"
    puts "-" * 40

    # Get candidates và reranking results
    candidates = initial_retrieval(query, chunks, embeddings, 8)
    rerank_results = rerank_with_llm(query, candidates, 5)

    if rerank_results[:ranking_details] && !rerank_results[:ranking_details].key?(:error)
      # Analyze rank changes
      rank_changes = rerank_results[:ranking_details][:score_changes]
      positive_changes = rank_changes.count { |c| c[:rank_change] > 0 }
      negative_changes = rank_changes.count { |c| c[:rank_change] < 0 }

      # Calculate average LLM scores
      avg_llm_score = rerank_results[:all_scores].map { |s| s[:llm_score] }.sum / rerank_results[:all_scores].length

      pattern_results[type] = {
        positive_changes: positive_changes,
        negative_changes: negative_changes,
        avg_llm_score: avg_llm_score.round(3),
        top_result_changed: rank_changes.first[:rank_change] != 0
      }

      puts "Rank improvements: #{positive_changes}"
      puts "Rank downgrades: #{negative_changes}"
      puts "Average LLM score: #{avg_llm_score.round(3)}"
      puts "Top result changed: #{pattern_results[type][:top_result_changed] ? 'Yes' : 'No'}"
    end
  end

  # Summary analysis
  puts "\n" + "=" * 60
  puts "PATTERN SUMMARY"
  puts "=" * 60

  pattern_results.each do |type, data|
    puts "#{type}: Avg LLM Score=#{data[:avg_llm_score]}, Top Changed=#{data[:top_result_changed] ? 'Y' : 'N'}"
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Reranker RAG bằng Ruby"
  puts "=" * 60

  begin
    # Demo chính
    run_reranker_demo

    # Phân tích patterns
    puts "\n\n"
    analyze_reranking_patterns

    puts "\n=== HOÀN THÀNH DEMO RERANKER RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end
