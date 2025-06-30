#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Demo RAG vs Non-RAG (Phiên bản Ruby)
# ===============================================================================
#
# Script này so sánh kết quả của việc sử dụng và không sử dụng RAG
# khi hỏi về 6 giá trị cốt lõi của Zigexn Ventura.
#
# Trong thiết lập demo này, chúng ta thực hiện:
#
# 1. Hỏi câu hỏi về 6 giá trị cốt lõi của Zigexn Ventura mà không có RAG
# 2. Hỏi câu hỏi tương tự nhưng sử dụng RAG với dữ liệu từ file Zigexn.txt
# 3. So sánh kết quả

require 'net/http'
require 'uri'
require 'json'
require 'matrix'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================

# Tùy chọn hiển thị
SHOW_DETAILED_OUTPUT = false # Đặt thành true để hiển thị các thông báo chi tiết

def puts_if_detailed(message)
  puts message if SHOW_DETAILED_OUTPUT
end

puts_if_detailed "=== Thiết lập Môi trường Demo RAG ==="

# Đọc file .env nếu tồn tại
if File.exist?('.env')
  File.readlines('.env').each do |line|
    key, value = line.strip.split('=', 2)
    ENV[key] = value if key && value
  end
end

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

# Kiểm tra xem API key có sẵn không
if OPENAI_API_KEY.nil? || OPENAI_API_KEY.empty?
  puts_if_detailed "Cảnh báo: Biến môi trường OPENAI_API_KEY chưa được đặt"
else
  puts_if_detailed "API key đã được tải thành công"
end

# ===============================================================================
# Chia nhỏ Văn bản
# ===============================================================================

def chunk_text(text, chunk_size, overlap)
  # Chia văn bản thành các đoạn có kích thước chunk_size ký tự với overlap.
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Các Hàm OpenAI API Client
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

def create_embeddings(text, model = EMBEDDING_MODEL)
  """
  Tạo embeddings cho văn bản sử dụng mô hình được chỉ định.
  """

  payload = {
    model: model,
    input: text
  }

  make_api_request('embeddings', payload)
end

def generate_response(system_prompt, user_message, model = CHAT_MODEL)
  """
  Tạo phản hồi từ mô hình AI dựa trên system prompt và user message.
  """

  payload = {
    model: model,
    temperature: 0,
    messages: [
      { role: 'system', content: system_prompt },
      { role: 'user', content: user_message }
    ]
  }

  make_api_request('chat/completions', payload)
end

# ===============================================================================
# Các Hàm Tìm kiếm Semantic
# ===============================================================================

def cosine_similarity(vec1, vec2)
  # Tính cosine similarity giữa hai vector.
  v1 = vec1.is_a?(Array) ? Vector[*vec1] : vec1
  v2 = vec2.is_a?(Array) ? Vector[*vec2] : vec2

  # Tính dot product và magnitudes
  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  # Trả về cosine similarity
  dot_product / (magnitude1 * magnitude2)
end

def semantic_search(query, text_chunks, embeddings, k = 5)
  # Thực hiện tìm kiếm semantic trên các đoạn văn bản sử dụng query và embeddings.

  # Tạo embedding cho query
  query_response = create_embeddings(query)
  return [] unless query_response && query_response['data']

  query_embedding = query_response['data'][0]['embedding']
  similarity_scores = []

  # Tính điểm similarity giữa query embedding và mỗi text chunk embedding
  embeddings.each_with_index do |chunk_embedding, i|
    similarity_score = cosine_similarity(query_embedding, chunk_embedding['embedding'])
    similarity_scores << [i, similarity_score]
  end

  # Sắp xếp điểm similarity theo thứ tự giảm dần và lấy top k
  top_indices = similarity_scores.sort_by { |_, score| -score }.first(k).map(&:first)

  # Trả về top k đoạn văn bản liên quan nhất
  top_indices.map { |index| text_chunks[index] }
end

# ===============================================================================
# Triển khai Demo
# ===============================================================================

def run_demo
  puts "\n=== Demo So sánh RAG vs Non-RAG ==="

  # Câu hỏi mẫu
  query = "List and explain the 6 core values of Zigexn Ventura company?"

  puts "\n=== PHẦN 1: KHÔNG SỬ DỤNG RAG ==="
  puts "Câu hỏi: #{query}"

  # Không sử dụng RAG - chỉ dùng mô hình ngôn ngữ
  system_prompt_no_rag = "Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ nhất có thể dựa trên kiến thức của bạn."

  no_rag_response = generate_response(system_prompt_no_rag, query)

  if no_rag_response && no_rag_response['choices']
    puts "\nPhản hồi khi KHÔNG sử dụng RAG:"
    puts "=" * 80
    puts no_rag_response['choices'][0]['message']['content']
    puts "=" * 80
  else
    puts "Không thể tạo phản hồi AI khi không sử dụng RAG"
    return
  end

  puts "\n=== PHẦN 2: SỬ DỤNG RAG ==="
  puts "Câu hỏi: #{query}"

  # Đọc dữ liệu từ file Zigexn.txt
  zigexn_file_path = "data/Zigexn.txt"

  if File.exist?(zigexn_file_path)
    zigexn_text = File.read(zigexn_file_path)
    puts_if_detailed "Đã đọc dữ liệu từ file Zigexn.txt (#{zigexn_text.length} ký tự)"
  else
    puts "Không tìm thấy file Zigexn.txt"
    return
  end

  # Chia nhỏ văn bản
  text_chunks = chunk_text(zigexn_text, 1000, 200)
  puts_if_detailed "Số lượng đoạn văn bản: #{text_chunks.length}"

  # Tạo embeddings
  embeddings_response = create_embeddings(text_chunks)

  unless embeddings_response && embeddings_response['data']
    puts "Không thể tạo embeddings. Vui lòng kiểm tra API key và kết nối."
    return
  end

  embeddings = embeddings_response['data']
  puts_if_detailed "Đã tạo #{embeddings.length} embeddings"

  # Thực hiện tìm kiếm semantic
  top_chunks = semantic_search(query, text_chunks, embeddings, 3)

  if SHOW_DETAILED_OUTPUT
    puts "\nCác đoạn liên quan nhất:"
    top_chunks.each_with_index do |chunk, i|
      puts "Ngữ cảnh #{i + 1}:"
      puts chunk
      puts "-" * 50
    end
  end

  # Tạo phản hồi với RAG
  system_prompt_rag = "Bạn là một trợ lý AI chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không thể được suy ra trực tiếp từ ngữ cảnh được cung cấp, hãy trả lời: 'Tôi không có đủ thông tin để trả lời câu hỏi đó.'"

  user_prompt = top_chunks.map.with_index do |chunk, i|
    "Ngữ cảnh #{i + 1}:\n#{chunk}\n#{'=' * 50}\n"
  end.join("\n")
  user_prompt += "\nCâu hỏi: #{query}"

  rag_response = generate_response(system_prompt_rag, user_prompt)

  if rag_response && rag_response['choices']
    puts "\nPhản hồi khi SỬ DỤNG RAG:"
    puts "=" * 80
    puts rag_response['choices'][0]['message']['content']
    puts "=" * 80
  else
    puts "Không thể tạo phản hồi AI khi sử dụng RAG"
    return
  end

  if SHOW_DETAILED_OUTPUT
    puts "\n=== KẾT LUẬN ==="
    puts "Bạn có thể thấy sự khác biệt giữa hai phản hồi:"
    puts "1. Phản hồi không có RAG: Dựa vào kiến thức có sẵn của mô hình, có thể không chính xác hoặc thiếu thông tin cụ thể."
    puts "2. Phản hồi có RAG: Dựa trên dữ liệu thực tế từ file Zigexn.txt, cung cấp thông tin chính xác và cụ thể."
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts_if_detailed 'Demo So sánh RAG vs Non-RAG bằng Ruby'
  puts_if_detailed '=' * 50

  run_demo
end
