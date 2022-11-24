# Databricks notebook source
dbutils.widgets.text("Message2", "")
text = dbutils.widgets.get("Message2")
print(text)
