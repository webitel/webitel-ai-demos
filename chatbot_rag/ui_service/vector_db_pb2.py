# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vector_db.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fvector_db.proto\x12\tvector_db\":\n\x07\x41rticle\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\t\x12\x12\n\ncategories\x18\x03 \x03(\t\":\n\x12\x41\x64\x64\x41rticlesRequest\x12$\n\x08\x61rticles\x18\x01 \x03(\x0b\x32\x12.vector_db.Article\";\n\x13\x41\x64\x64\x41rticlesResponse\x12\n\n\x02id\x18\x01 \x03(\t\x12\x18\n\x10response_message\x18\x02 \x01(\t\"#\n\x15RemoveArticlesRequest\x12\n\n\x02id\x18\x01 \x03(\t\">\n\x16RemoveArticlesResponse\x12\n\n\x02id\x18\x01 \x03(\t\x12\x18\n\x10response_message\x18\x02 \x01(\t\"4\n\x12GetArticlesRequest\x12\n\n\x02id\x18\x01 \x03(\t\x12\x12\n\ncategories\x18\x02 \x03(\t\";\n\x13GetArticlesResponse\x12$\n\x08\x61rticles\x18\x01 \x03(\x0b\x32\x12.vector_db.Article\"=\n\x15UpdateArticlesRequest\x12$\n\x08\x61rticles\x18\x01 \x03(\x0b\x32\x12.vector_db.Article\"2\n\x16UpdateArticlesResponse\x12\x18\n\x10response_message\x18\x01 \x01(\t2\xdb\x02\n\x0fVectorDBService\x12L\n\x0b\x41\x64\x64\x41rticles\x12\x1d.vector_db.AddArticlesRequest\x1a\x1e.vector_db.AddArticlesResponse\x12L\n\x0bGetArticles\x12\x1d.vector_db.GetArticlesRequest\x1a\x1e.vector_db.GetArticlesResponse\x12U\n\x0eRemoveArticles\x12 .vector_db.RemoveArticlesRequest\x1a!.vector_db.RemoveArticlesResponse\x12U\n\x0eUpdateArticles\x12 .vector_db.UpdateArticlesRequest\x1a!.vector_db.UpdateArticlesResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vector_db_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ARTICLE']._serialized_start=30
  _globals['_ARTICLE']._serialized_end=88
  _globals['_ADDARTICLESREQUEST']._serialized_start=90
  _globals['_ADDARTICLESREQUEST']._serialized_end=148
  _globals['_ADDARTICLESRESPONSE']._serialized_start=150
  _globals['_ADDARTICLESRESPONSE']._serialized_end=209
  _globals['_REMOVEARTICLESREQUEST']._serialized_start=211
  _globals['_REMOVEARTICLESREQUEST']._serialized_end=246
  _globals['_REMOVEARTICLESRESPONSE']._serialized_start=248
  _globals['_REMOVEARTICLESRESPONSE']._serialized_end=310
  _globals['_GETARTICLESREQUEST']._serialized_start=312
  _globals['_GETARTICLESREQUEST']._serialized_end=364
  _globals['_GETARTICLESRESPONSE']._serialized_start=366
  _globals['_GETARTICLESRESPONSE']._serialized_end=425
  _globals['_UPDATEARTICLESREQUEST']._serialized_start=427
  _globals['_UPDATEARTICLESREQUEST']._serialized_end=488
  _globals['_UPDATEARTICLESRESPONSE']._serialized_start=490
  _globals['_UPDATEARTICLESRESPONSE']._serialized_end=540
  _globals['_VECTORDBSERVICE']._serialized_start=543
  _globals['_VECTORDBSERVICE']._serialized_end=890
# @@protoc_insertion_point(module_scope)
