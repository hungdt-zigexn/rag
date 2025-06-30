# frozen_string_literal: true

source 'https://rubygems.org'

ruby '>= 3.0.0'

# Core dependencies for RAG implementation
gem 'json', '~> 2.6'
gem 'matrix', '~> 0.4'

# HTTP client for API requests
gem 'httparty', '~> 0.21'

# PDF processing (optional, for production use)
gem 'pdf-reader', '~> 2.11'

# Development and testing
group :development, :test do
  gem 'rspec', '~> 3.12'
  gem 'rubocop', '~> 1.56'
  gem 'pry', '~> 0.14'
end

# Optional gems for enhanced functionality
group :optional do
  gem 'parallel', '~> 1.23' # For parallel processing
  gem 'redis', '~> 5.0'     # For caching embeddings
  gem 'sqlite3', '~> 1.6'   # For storing embeddings locally
end 