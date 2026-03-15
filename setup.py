"""Setup configuration for Code Evaluator package"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_path, 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='code-evaluator',
    version='1.0.0',
    author='VanAnh',
    author_email='your-email@example.com',
    description='Multi-language code analysis platform using LLM API providers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VanAnh-13/code_evaluator',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'docs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Testing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'flask>=2.3.3',
        'werkzeug>=2.3.7',
        'flask-wtf>=1.2.1',
        'wtforms>=3.0.1',
        'python-dotenv>=1.0.0',
        'colorama>=0.4.6',
        'tqdm>=4.66.1',
        'requests>=2.31.0',
    ],
    extras_require={
        'openai': ['openai>=1.30.0'],
        'anthropic': ['anthropic>=0.26.0'],
        'gemini': ['google-generativeai>=0.5.0'],
        'all': [
            'openai>=1.30.0',
            'anthropic>=0.26.0',
            'google-generativeai>=0.5.0',
        ],
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'pytest-mock>=3.12.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'code-evaluator=code_evaluator.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'code_evaluator': [
            'prompts/*.txt',
            'prompts/*.json',
            'web/templates/*.html',
            'web/static/css/*.css',
            'web/static/js/*.js',
        ],
    },
    keywords='code analysis linting quality llm ai testing',
    project_urls={
        'Bug Reports': 'https://github.com/VanAnh-13/code_evaluator/issues',
        'Source': 'https://github.com/VanAnh-13/code_evaluator',
        'Documentation': 'https://github.com/VanAnh-13/code_evaluator/blob/master/README.md',
    },
)
