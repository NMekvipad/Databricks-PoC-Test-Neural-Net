# Databricks notebook source
dbutils.widgets.text("Message1", "")
text = dbutils.widgets.get("Message1")
print(text)
